#pragma once
#include "layer.h"
#include "traindata.h"
namespace nn {
	class costfunction :
		public Layer
	{
	public:
		costfunction(string name = "");
		~costfunction();

		void setfunction(LossFunc loss_f);
		void setfunction(ReduceType loss_f);

		Size3 initialize(Size3 param_size);
		void forword(const Mat &in, Mat &out)const;
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable);
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const;	
		void forword(const Mat &label, const Mat &output, Mat &dst)const;
		float forword(TrainData::iterator label, const vector<vector<Mat>> *output, size_t idx)const;
		void back(const Mat &label, const Mat &output, Mat &dst)const;
		void back(TrainData::iterator label, vector<vector<Mat>> *output, size_t idx)const;
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file);
		void show(std::ostream &out)const;

		LossFunc f;
		LossFunc df;
		bool ignore_active = false;
		float weight = 1.0f;
	};
}