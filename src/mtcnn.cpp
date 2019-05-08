#include "MTCNN.h"
#include "include.h"
#include <iostream>
#include <windows.h>
using std::cout;
using std::endl;


nn::MTCNN::MTCNN()
{
}


nn::MTCNN::~MTCNN()
{
}

void nn::MTCNN::detect(Mat & img_, std::vector<Bbox>& finalBbox_)
{
	img = img_;
	img_w = img.cols();
	img_h = img.rows();
	img = (img - 127.5f)*0.0078125f;

	PNet();
	//the first stage's nms
	if (firstBbox_.empty()) return;
	method::nms(firstBbox_, nms_threshold[0]);
	method::refine(firstBbox_, img_h, img_w, true);
	//printf("firstBbox_.size()=%d\n", firstBbox_.size());
	firstBbox_.swap(finalBbox_);
	return;
	//second stage
	RNet();
	//printf("secondBbox_.size()=%d\n", secondBbox_.size());
	if (secondBbox_.size() < 1) return;
	method::nms(secondBbox_, nms_threshold[1]);
	method::refine(secondBbox_, img_h, img_w, true);

	//third stage 
	ONet();
	//printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
	if (thirdBbox_.empty()) return;
	method::refine(thirdBbox_, img_h, img_w, true);
	method::nms(thirdBbox_, nms_threshold[2], "Min");
	thirdBbox_.swap(finalBbox_);
}

void nn::MTCNN::PNet()
{
	firstBbox_.clear();
	float minl = float(img_w < img_h ? img_w : img_h);
	float m = (float)MIN_DET_SIZE / minsize;
	minl *= m;
	float factor = pre_facetor;
	vector<float> scales_;
	while (minl > MIN_DET_SIZE) {
		scales_.push_back(m);
		minl *= factor;
		m = m * factor;
	}
	for (size_t i = 0; i < scales_.size(); i++) {
		int hs = (int)ceil(img_h * scales_[i]);
		int ws = (int)ceil(img_w * scales_[i]);
		Mat in;
		Mat score_, location_;
		method::resize(img, in, Size(hs, ws), LocalMean);
		vector<Mat> out = Pnet(in);
		score_ = out[0];
		location_ = out[1];
		std::vector<Bbox> boundingBox_;
		method::generateBbox(score_, location_, boundingBox_, scales_[i], threshold[0]);
		method::nms(boundingBox_, nms_threshold[0]);
		firstBbox_.insert(firstBbox_.end(), boundingBox_.begin(), boundingBox_.end());
		boundingBox_.clear();
	}
}

void nn::MTCNN::RNet()
{
	secondBbox_.clear();
	int count = 0;
	for (vector<Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++) {
		Mat tempIm;
		tempIm = copyMakeBorder(img, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
		Mat in, score, bbox;
		method::resize(tempIm, in, 24, 24, LocalMean);
		vector<Mat> out = Pnet(in);
		score = out[0];
		bbox = out[1];
		if (score(0, 0, 1) > threshold[1]) {
			for (int channel = 0; channel < 4; channel++) {
				it->regreCoord[channel] = bbox(0, 0, channel);
			}
			it->area = float((it->x2 - it->x1) * (it->y2 - it->y1));
			it->score = score(0, 0, 1);
			secondBbox_.push_back(*it);
		}
	}
}

void nn::MTCNN::ONet()
{
	thirdBbox_.clear();
	for (vector<Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++) {
		Mat tempIm;
		tempIm = copyMakeBorder(img, (*it).y1, img_h - (*it).y2, (*it).x1, img_w - (*it).x2);
		Mat in;
		method::resize(tempIm, in, 48, 48, LocalMean);
		Mat score, bbox, keyPoint;
		vector<Mat> out = Pnet(in);
		score = out[0];
		bbox = out[1];
		keyPoint = out[1];
		if (score(0, 0, 1) > threshold[2]) {
			for (int channel = 0; channel < 4; channel++) {
				it->regreCoord[channel] = bbox(0, 0, channel);;
			}
			it->area = float((it->x2 - it->x1) * (it->y2 - it->y1));
			it->score = score(0, 0, 1);
			for (int num = 0; num < 5; num++) {
				(it->ppoint)[num] = it->x1 + (it->x2 - it->x1) * keyPoint(0, 0, num);
				(it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1) * keyPoint(0, 0, num + 5);
			}
			thirdBbox_.push_back(*it);
		}
	}
}

const Mat nn::MTCNN::FaceClassifyLoss(const Mat &y, const Mat &y0)
{
	Mat y1 = zeros(y0.size3());
	if (y(0) == -1.0f)
		return y1;
	else {
		y1((int)y(0)) = 1.0f;
		return CrossEntropy(y1, y0);
	}
}

const Mat nn::MTCNN::FaceClassifyLoss_D(const Mat &y, const Mat &y0)
{
	Mat y1 = zeros(y0.size3());
	if (y(0) == -1.0f)
		return y1;
	else {
		y1((int)y(0)) = 1.0f;
		return D_CrossEntropy(y1, y0);
	}
}

const Mat nn::MTCNN::BboxLoss(const Mat &y, const Mat &y0)
{
	if (y.sum() == 0.0f)
		return y;
	else
		return L2(y, y0);
}

const Mat nn::MTCNN::BboxLoss_D(const Mat &y, const Mat &y0)
{
	if (y.sum() == 0.0f)
		return y;
	else
		return D_L2(y, y0);
}

void nn::MTCNN::image_processing(const Mat& src, Mat&dst)
{
	dst = (src - 127.5f)*0.0078125f;
}

const vector<Mat> nn::MTCNN::label_processing(const Mat &label)
{
	//vector<Mat> label_(1, value(label(0), 1, 1, 1));
	vector<Mat> label_(2, value(label(0), 1, 1, 1));
	if (label(0) == 0.0f)
		label_[1] = zeros(1, 1, 4);
	else {
		label_[1] = Block(label, 1, label.rows() - 1, 0, 0);
		label_[1].reshape(1, 1, 4);
		/*float temp = label_[1](0);
		label_[1](0) = -label_[1](2);
		label_[1](2) = -temp;*/
	}
	return label_;
}

nn::Net nn::MTCNN::create_pnet(bool load, string model)
{
	Size3 size(12, 12, 3); 
	Net net;
	net.add(Conv2D(10, 3, false, 0, "conv_1"));
	net.add(PReLU("activate_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, 0, "conv_2"));
	net.add(PReLU("activate_2"));
	net.add(Conv2D(32, 3, false, 0, "conv_3"));
	net.add(PReLU("activate_3"));
	//net.add(Conv2D(2, 1, false, Softmax, "fcl"));
	//net.add(Loss(CrossEntropy, 1, "fc_loss"));
	net.add(Conv2D(2, 1, false, Softmax, "fc1"));
	net.add(Conv2D(4, 1, false, 0, "bbox"), "activate_3", true);
	net.add(Loss(FaceClassifyLoss, FaceClassifyLoss_D, true, 1, "fc_loss"));
	net.add(Loss(BboxLoss, BboxLoss_D, false, 0, "bbox_loss"), "bbox");
	net.initialize(size);
	if(load)
		net.load_param(model);
	return net;
}

void nn::MTCNN::trainPNet(string rootpath, string imglist, string imagedir,
	int batch_size, bool load, string model, string optimizer_param)
{
	Size3 size(12, 12, 3);
	Pnet = create_pnet(load, model);

	cout << Pnet << endl;
	Mat x = mRand(-1, 1, size, true);

	vector<Mat> output = Pnet(x);
	cout << "input " << x.size3() << endl;
	cout << "output ";
	for (Mat &y : output)
		cout << y.size3() << endl;

	Train train;
	train.regularization = true;
	train.lambda = 0.001f;
	Optimizer *optimizer = Optimizer::CreateOptimizer(OptimizerInfo(Adam, 1e-3f));
	if (load) {
		optimizer->load(optimizer_param);
	}
	TrainData trainData(rootpath,
		imglist, imagedir,
		batch_size, label_processing);
	trainData.register_process(0, image_processing);
	trainData.load_all_data(true);

	if (load) {
		train.RegisterNet(&Pnet);
		train.RegisterOptimizer(optimizer);
		train.initialize();
	}
	else
		train.Fit(&Pnet, trainData, optimizer, 1, 1, true, true);

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	int count, recount;
	count = recount = 0;
	int success, sum, background, human, class_1, class_2;
	while (1) {
		if (trainData.all_load()) {
			success = sum = background = human = class_1 = class_2 = 0;
			TrainData::Databox boxes = trainData.all_batches();
			for (TrainData::DataboxIter &vec : *boxes) {
				for (const NetData *data : vec) {
					if (data->label[0](0) == -1.0f)continue;
					Mat output = Pnet(data->input)[0];
					if (output.maxAt() == (int)data->label[0](0))
						success += 1;
					if ((int)data->label[0](0) == 1) {
						human += 1;
					}
					else {
						background += 1;
					}
					if (output.maxAt() == 1) {
						class_2 += 1;
					}
					else {
						class_1 += 1;
					}
					sum += 1;
				}
			}
			printf("background(%d, %d), human(%d, %d), total(%d, %d), acc %0.4f\n", class_1, background, class_2, human, success, sum, (float)success / (float)sum * 100);
		}
		/*Image img = Imread("D:\\mtcnn\\test01.jpg");
		if (!img.empty()) {
			Mat m = Image2Mat(img);
			vector<Bbox> finalBbox;
			QueryPerformanceCounter(&t1);
			detect(m, finalBbox);
			QueryPerformanceCounter(&t2);
			cout << finalBbox.size() << " cost time: " << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << "sec" << endl;
			for (Bbox bbox : finalBbox)
			{
				rectangle(img, bbox.x1, bbox.y1, bbox.x2, bbox.y2, Color(rand() % 256, rand() % 256, rand() % 256), 1);
			}

			Imwrite("./test.jpg", img);
			img.release();
			m.release();
			finalBbox.clear();
		}*/
		train.Fit(trainData, 1, 1, false);
		printf("count: %d\n", count + 1);
		Pnet.save_param(model);
		optimizer->save(optimizer_param);
		count += 1;
		//count = (count + 1) % 100;
		//if (count == 99)
		//{
		//	//optimizer->Step() = optimizer->Step()*0.1;
		//	recount += 1;
		//}
		//if (recount == 4)break;
	}
	delete optimizer;
	optimizer = nullptr;
}


void nn::MTCNN::testPNet(string model, string pic, string savepath)
{
	Pnet = create_pnet(true, model);

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	Image img = Imread(pic);
	if (!img.empty()) {
		Mat m = Image2Mat(img);
		vector<Bbox> finalBbox;
		QueryPerformanceCounter(&t1);
		detect(m, finalBbox);
		QueryPerformanceCounter(&t2);
		cout << finalBbox.size() << " cost time: " << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << "sec" << endl;
		for (Bbox bbox : finalBbox)
		{
			//if(bbox.score>0.9f)
			rectangle(img, bbox.x1, bbox.y1, bbox.x2, bbox.y2, Color(rand() % 256, rand() % 256, rand() % 256), 1);
		}

		Imwrite(savepath, img);
		img.release();
		m.release();
		finalBbox.clear();
	}
}