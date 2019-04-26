#include "MTCNN.h"

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
	nms(firstBbox_, nms_threshold[0]);
	refine(firstBbox_, img_h, img_w, true);
	//printf("firstBbox_.size()=%d\n", firstBbox_.size());
	firstBbox_.swap(finalBbox_);
	return;
	//second stage
	RNet();
	//printf("secondBbox_.size()=%d\n", secondBbox_.size());
	if (secondBbox_.size() < 1) return;
	nms(secondBbox_, nms_threshold[1]);
	refine(secondBbox_, img_h, img_w, true);

	//third stage 
	ONet();
	//printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
	if (thirdBbox_.empty()) return;
	refine(thirdBbox_, img_h, img_w, true);
	nms(thirdBbox_, nms_threshold[2], "Min");
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
		resize(img, in, Size(hs, ws), LocalMean);
		vector<Mat> out = Pnet(in);
		score_ = out[0];
		location_ = out[1];
		std::vector<Bbox> boundingBox_;
		generateBbox(score_, location_, boundingBox_, scales_[i], threshold[0]);
		nms(boundingBox_, nms_threshold[0]);
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
		resize(tempIm, in, 24, 24, LocalMean);
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
		resize(tempIm, in, 48, 48, LocalMean);
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
