#include <iostream>
#include <fstream>
#include <windows.h>
#include "include.h"
#include "mtcnn.h"
#include "json.hpp"

using namespace nn;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;


void reverse(const Image& src, Image&dst)
{
	/*if (src.channels != 1)
		RGB2Gray(src, dst);
	else
		dst = src;*/
	dst = 255 - src;
}

void normalization(const Mat& src, Mat&dst)
{
	dst = src / src.findmax();
}

const vector<Mat> label(const Mat &label)
{
	return vector<Mat>(1, value((float)label.maxAt(), 1, 1, 1));
}

void testNet()
{
	Net net;
	net.add(Conv2D(10, 3, false, ReLU, "conv_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, ReLU, "conv_2"));
	net.add(Conv2D(32, 3, false, ReLU, "conv_3")); 
	net.add(Conv2D(2, 1, false, Softmax, "fcl"));
	net.add(Loss(CROSSENTROPY, 1, "fc_loss"));
	net.add(Conv2D(4, 1, false, 0, "bbox"), "fcl", true);
	net.add(Loss(L2, 1, "bbox_loss"), "bbox");
	net.add(Conv2D(10, 1, false, 0, "landmark"), "fcl", true);
	net.add(Loss(L2, 1, "landmark_loss"), "landmark");
	net.initialize(Size3(12, 12, 3));
	cout << net << endl;
	Mat x = mRand(-1, 1, 12, 12, 3, true);
	vector<Mat> output = net(x);
	for(Mat &y : output)
		cout << y << endl;
}
const Mat ReadImageData(const Image &input)
{
	Image dst = 255 - input;//黑白转换
	Mat mat = Image2Mat(dst);//图像转为矩阵
	mat /= Max(mat);//归一化
	return mat;
}
void testword(Net &net)
{
	vector<string> files;
	getFiles("F:\\NeuralNetworks\\train\\images", files);//保存所有文件路径
	if (files.empty()) {
		printf("file is empty!\n");
		return;
	}
	string word = "ABCDEFGHJKLMNPRSTUVWXYZ";//所有结果
	int success = 0;
	int sum = 0;
	for (string &file : files) {
		size_t right = file.rfind('\\');
		size_t left = file.rfind('\\', right - 1) + 1;
		char key = file.substr(left, right - left)[0];
		Image image = Imread(file.c_str());//用灰度读取图像
		vector<Mat> output = net(ReadImageData(image));//运行网络输出
		int adr = output[0].maxAt();//取最大值索引
		if (word[adr] == key)
			success += 1;
		sum += 1;
		cout << word[adr] << ' ' << adr << ' ' << key << '\t';
		cout << file << endl;
	}
	printf("样本正确率 %lf\n", float(success) / float(sum));
}
void testTrainNet()
{
	Size3 size(16, 16, 3);
	Net net;
	net.add(Conv2D(10, 3, false, 0, "conv_1"));
	net.add(BatchNorm("bn_1"));
	net.add(PReLU("activate_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, 0, "conv_2"));
	net.add(BatchNorm("bn_2"));
	net.add(PReLU("activate_2"));
	net.add(Conv2D(32, 3, false, 0, "conv_3"));
	net.add(BatchNorm("bn_3"));
	net.add(PReLU("activate_3"));
	net.add(Conv2D(23, 3, false, 0, "conv_4"));
	net.add(Reshape(Size3(23, 1, 1), "output"));
	net.add(Loss(SoftmaxCrossEntropy, 1, "loss"));
	net.initialize(size);
	cout << net << endl;
	Mat x = mRand(-1, 1, size, true);
	vector<Mat> output = net(x);
	cout << "input " << x.size3() << endl;
	cout << "output ";
	for (Mat &y : output)
		cout << y.size3() << endl;

	int epoch = 20;
	TrainData trainData("F:\\NeuralNetworks\\train", "data.txt", "images", 128);
	trainData.register_process(reverse, normalization);
	cout << "load traindata..." << endl;
	trainData.load_all_data(true);
	printf("start train net...\n");

	Train train; 
	//train.regularization = true;
	//train.lambda = 0.001f;
	train.Fit(&net, trainData, OptimizerInfo(Momentum, 0.0001f), epoch, 1, true);
	net.save("./model/cnn");
	net.clear();
	net.load("./model/cnn");
	testword(net);
}

const Mat FaceClassifyLoss(const Mat &y, const Mat &y0)
{
	Mat y1 = zeros(y0.size3());
	if (y(0) == -1.0f)
		return y1;
	else {
		y1((int)y(0)) = 1.0f;
		return CrossEntropy(y1, y0);
	}
}

const Mat FaceClassifyLoss_D(const Mat &y, const Mat &y0)
{
	Mat y1 = zeros(y0.size3());
	if (y(0) == -1.0f)
		return y1;
	else {
		y1((int)y(0)) = 1.0f;
		return D_CrossEntropy(y1, y0);
	}
}

const Mat BboxLoss(const Mat &y, const Mat &y0)
{
	if (y.sum() == 0.0f)
		return y;
	else 
		return L2(y, y0);
}

const Mat BboxLoss_D(const Mat &y, const Mat &y0)
{
	if (y.sum() == 0.0f)
		return y;
	else 
		return D_L2(y, y0);
}

void image_processing(const Mat& src, Mat&dst)
{
	dst = (src - 127.5f)*0.0078125f;
}

const vector<Mat> label_processing(const Mat &label)
{
	//vector<Mat> label_(1, value(label(0), 1, 1, 1));
	vector<Mat> label_(2, value(label(0), 1, 1, 1));
	if (label(0) == 0.0f)
		label_[1] = zeros(1, 1, 4);
	else {
		label_[1] = Block(label, 1, label.rows() - 1, 0, 0);
		label_[1].reshape(1, 1, 4);
		//float temp = label_[1](0);
		//label_[1](0) = -label_[1](2);
		//label_[1](2) = -temp;
	}
	return label_;
}
void testSSD()
{
	Size3 size(256, 256, 3);
	MTCNN mtcnn;
	mtcnn.Pnet.add(Conv2D(10, 3, false, ReLU, "conv_1"));
	mtcnn.Pnet.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	mtcnn.Pnet.add(Conv2D(16, 3, false, ReLU, "conv_2"));
	mtcnn.Pnet.add(Conv2D(32, 3, false, ReLU, "conv_3"));
	//mtcnn.Pnet.add(Conv2D(2, 1, false, Softmax, "fcl"));
	mtcnn.Pnet.add(Conv2D(2, 1, false, Softmax, "fc1"));
	//mtcnn.Pnet.add(Loss(CrossEntropy, 1, "fc_loss"));
	mtcnn.Pnet.add(Loss(FaceClassifyLoss, FaceClassifyLoss_D, true, 1, "fc_loss"));
	mtcnn.Pnet.add(Conv2D(4, 1, false, 0, "bbox"), "conv_3", true);
	mtcnn.Pnet.add(Loss(BboxLoss, BboxLoss_D, false, 1, "bbox_loss"), "bbox");
	mtcnn.Pnet.initialize(size);
}
void testPNet()
{
//#define LOAD_DATA
	Size3 size(12, 12, 3);
	MTCNN mtcnn;
	mtcnn.Pnet.add(Conv2D(10, 3, false, 0, "conv_1"));
	mtcnn.Pnet.add(PReLU("activate_1"));
	mtcnn.Pnet.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	mtcnn.Pnet.add(Conv2D(16, 3, false, 0, "conv_2"));
	mtcnn.Pnet.add(PReLU("activate_2"));
	mtcnn.Pnet.add(Conv2D(32, 3, false, 0, "conv_3"));
	mtcnn.Pnet.add(PReLU("activate_3"));
	//mtcnn.Pnet.add(Conv2D(2, 1, false, Softmax, "fcl"));
	mtcnn.Pnet.add(Conv2D(2, 1, false, Softmax, "fc1"));
	//mtcnn.Pnet.add(Loss(CrossEntropy, 1, "fc_loss"));
	mtcnn.Pnet.add(Loss(FaceClassifyLoss, FaceClassifyLoss_D, true, 1, "fc_loss"));
	mtcnn.Pnet.add(Conv2D(4, 1, false, 0, "bbox"), "conv_3", true);
	mtcnn.Pnet.add(Loss(BboxLoss, BboxLoss_D, false, 0.5, "bbox_loss"), "bbox");
	mtcnn.Pnet.initialize(size);
#ifdef LOAD_DATA
	mtcnn.Pnet.load_param("./net/net.param");
#endif // LOAD_DATA

	cout << mtcnn.Pnet << endl;
	Mat x = mRand(-1, 1, size, true);

	vector<Mat> output = mtcnn.Pnet(x);
	cout << "input " << x.size3() << endl;
	cout << "output ";
	for (Mat &y : output)
		cout << y.size3() << endl;

	Train train;
	//train.regularization = true;
	//train.lambda = 0.001f;
	Optimizer *optimizer = Optimizer::CreateOptimizer(OptimizerInfo(Adam, 1e-3f));
#ifdef LOAD_DATA
	optimizer->load("./net/optimizer.param");
#endif // LOAD_DATA
	TrainData trainData("F:\\deeplearn\\data\\seanlx", 
		"train_12.txt", "mtcnn1", 
		256, label_processing);
	trainData.register_process(0, image_processing);
	trainData.load_all_data(true);

#ifdef LOAD_DATA
	train.RegisterNet(&mtcnn.Pnet);
	train.RegisterOptimizer(optimizer);
	train.initialize();
#else
	train.Fit(&mtcnn.Pnet, trainData, optimizer, 1, 1, true, true);
#endif // LOAD_DATA

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	int success, sum;
	int count, recount, background, human, class_1, class_2;
	count = recount = 0;
	while (1) {
		success = sum = background = human = class_1 = class_2 = 0;
		TrainData::Databox boxes = trainData.all_batches();
		for (TrainData::DataboxIter &vec : *boxes) {
			for (const NetData *data : vec) {
				if (data->label[0](0) == -1.0f)continue;
				Mat output = mtcnn.Pnet(data->input)[0];
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
	
		Image img = Imread("./5ae1378_0.jpg");
		if (!img.empty()) {
			Mat m = Image2Mat(img);
			vector<Bbox> finalBbox;
			QueryPerformanceCounter(&t1);
			mtcnn.detect(m, finalBbox);
			QueryPerformanceCounter(&t2);
			cout << finalBbox.size() << " cost time: " << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << "sec" << endl;
			for (Bbox bbox : finalBbox)
			{
				//if(bbox.score>0.9f)
				rectangle(img, bbox.x1, bbox.y1, bbox.x2, bbox.y2, Color(rand() % 256, rand() % 256, rand() % 256), 1);
			}

			Imwrite("./test.jpg", img);
			img.release();
			m.release();
			finalBbox.clear();
		}

		train.Fit(trainData, 1, 1, false);
		printf("count: %d\n", count + 1);
		mtcnn.Pnet.save_param("./net/net.param");
		optimizer->save("./net/optimizer.param");
		count += 1;
		//count = (count + 1) % 100;
		//if (count == 99)
		//{
		//	//optimizer->Step() = optimizer->Step()*0.1;
		//	recount += 1;
		//}
		//if (recount == 4)break;
	}

	//for (int &v : trainData.range) {
	//	const vector<NetData> *netData = trainData.batches();
	//	for (const NetData &data : *netData) {
	//		cout << data.input.Sum() << " " << data.label[0] << " " << data.label[1].t() << endl;
	//	}
	//}


	//TrainData trainData("./train/face", "train.txt", "images", 3, label_processing);
	//trainData.register_process(0, image_processing);
	//for (int &v : trainData.range) {
	//	const vector<NetData> *netData = trainData.batches();
	//	for (const NetData &data : *netData) {
	//		cout << data.input.Sum() << " " << data.label[0] << " " << data.label[1].t() << endl;
	//	}
	//	train.Fit(*netData);
	//}

}
void PNet()
{
	Size3 size(12, 12, 3);
	MTCNN mtcnn;
	mtcnn.Pnet.add(Conv2D(10, 3, false, 0, "conv_1"));
	mtcnn.Pnet.add(PReLU("activate_1"));
	mtcnn.Pnet.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	mtcnn.Pnet.add(Conv2D(16, 3, false, 0, "conv_2"));
	mtcnn.Pnet.add(PReLU("activate_2"));
	mtcnn.Pnet.add(Conv2D(32, 3, false, 0, "conv_3"));
	mtcnn.Pnet.add(PReLU("activate_3"));
	//mtcnn.Pnet.add(Conv2D(2, 1, false, Softmax, "fcl"));
	mtcnn.Pnet.add(Conv2D(2, 1, false, Softmax, "fc1"));
	//mtcnn.Pnet.add(Loss(CrossEntropy, 1, "fc_loss"));
	mtcnn.Pnet.add(Loss(FaceClassifyLoss, FaceClassifyLoss_D, true, 1, "fc_loss"));
	mtcnn.Pnet.add(Conv2D(4, 1, false, 0, "bbox"), "conv_3", true);
	mtcnn.Pnet.add(Loss(BboxLoss, BboxLoss_D, false, 0.5, "bbox_loss"), "bbox");
	mtcnn.Pnet.initialize(size);
	mtcnn.Pnet.load_param("./net/net.param");

	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	Image img = Imread("F:\\NeuralNetworks\\1 (2).jpg");
	if (!img.empty()) {
		Mat m = Image2Mat(img);
		vector<Bbox> finalBbox;
		QueryPerformanceCounter(&t1);
		mtcnn.detect(m, finalBbox);
		QueryPerformanceCounter(&t2);
		cout << finalBbox.size() << " cost time: " << (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart << "sec" << endl;
		for (Bbox bbox : finalBbox)
		{
			//if(bbox.score>0.9f)
			rectangle(img, bbox.x1, bbox.y1, bbox.x2, bbox.y2, Color(rand() % 256, rand() % 256, rand() % 256), 1);
		}

		Imwrite("./test1.jpg", img);
		img.release();
		m.release();
		finalBbox.clear();
	}
}
void testTrain()
{
	TrainData traindata("D:\\mtcnn_data\\data\\dataset", "imglists/train_12.txt", "images", 1024, label_processing);
	traindata.register_process(0, image_processing);
	traindata.load_all_data(true);
	for (int &v : traindata.range) {
		TrainData::iterator netData = traindata.batches();
		for (const NetData *data : *netData) {
			cout << data->input.sum() << " " << data->label[0] << endl;
		}
	}
}
int main()
{
	Srandom();
	//testTrain();
	//testNet(); 
	//Net net;
	//net.load("./model/cnn");
	//testword(net);
	testTrainNet();
	/*vector<string> files;
	getFiles("E:\\deeplearn\\data\\kaggle_cifar10\\train", files);
	int count = 0;
	ofstream out("./negative.txt");
	for (string file : files)
	{
		Image img = Imread(file);
		resize(img, img, Size(12, 12), EqualIntervalSampling);
		Imwrite("D:\\mtcnn_data\\data\\dataset\\images\\12\\negative\\" + std::to_string(count+1) + ".jpg", img);
		count += 1;
		out << "12/negative/" + std::to_string(count + 1) + ".jpg " << 0 << endl;
		if (count > 10000)break;
	}out.close();*/
	//PNet();
	//testPNet();
	pause();
	return 0;
}