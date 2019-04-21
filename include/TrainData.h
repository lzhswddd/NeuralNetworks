#ifndef __TRAINDATA_H__
#define __TRAINDATA_H__

#include <string>
#include "Image.h"
#include "NetParam.h"

using std::vector;
using std::string;

namespace nn {
	class TrainData
	{
	public:
		TrainData();
		TrainData(string rootpath, string imglist, string imagedir,
			int batch_size, const vector<Mat>(*label_process)(const Mat&) = nullptr);
		void load_train_data(string rootpath, string imglist, string imagedir,
			int batch_size, const vector<Mat>(*label_process)(const Mat&) = nullptr);
		~TrainData();
		void reset();
		void clear();
		int batchSize()const;
		int len()const;
		const vector<NetData>* batches();
		void load_all_data();
		void register_process(void(*image)(const Image&, Image&) = nullptr, void(*mat)(const Mat&, Mat&) = nullptr);
		vector<int> range;
	protected:
		void next();
		void get_train_data(string imglist, string imagepath, const vector<Mat>(*label_proces)(const Mat&) = nullptr);
		const vector<Mat> labelProcess(const Mat& l);
	private:
		int index = 0;
		int batch_size;
		int batch_number;
		string rootpath;
		vector<Mat> data;
		vector<vector<Mat>> label;
		vector<string> imgpath;
		vector<vector<NetData>> batchdata;
		void(*process_image)(const Image&, Image&) = nullptr;
		void(*process_mat)(const Mat&, Mat&) = nullptr;
	};
}
#endif //__TRAINDATA_H__
