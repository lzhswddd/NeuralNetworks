#ifndef __TRAINDATA_H__
#define __TRAINDATA_H__

#include <string>
#include <vector>
#include "Mat.h"
#include "Image.h"

using std::vector;
using std::string;

namespace nn {
	class TrainData
	{
	public:
		TrainData();
		TrainData(string rootpath, string imglist, string imagedir,
			int batch_size, const Mat(*label_proces)(const Mat&) = nullptr);
		void load_train_data(string rootpath, string imglist, string imagedir,
			int batch_size, const Mat(*label_proces)(const Mat&) = nullptr);
		~TrainData();
		void reset();
		void clear();
		int len()const;
		void batch(Mat &x, Mat &y);
		void batches(vector<Mat> &x, vector<Mat> &y);
		void loadAllData();
		void register_process(void(*image)(const Image&, Image&) = nullptr, void(*mat)(const Mat&, Mat&) = nullptr);
		vector<int> range;
	protected:
		void next();
		void get_train_data(string imglist, string imagepath, const Mat(*label_proces)(const Mat&) = nullptr);
	private:
		int index = 0;
		int batch_size;
		string rootpath;
		vector<string> imgpath;
		vector<Mat> data;
		vector<Mat> label;
		void(*process_image)(const Image&, Image&) = nullptr;
		void(*process_mat)(const Mat&, Mat&) = nullptr;
	};
}
#endif //__TRAINDATA_H__
