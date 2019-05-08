#pragma once
#include "layer.h"

namespace nn {
	class batchnormalization :
		public parametrics
	{
	public:
		batchnormalization(string name = "");
		~batchnormalization();
		Size3 initialize(Size3 param_size);
		void updateregular();
		void update(const vector<Mat> &d, int *idx);
		void forword(const Mat &param, Mat &out)const;
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable);
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer, int *number)const;
		void append_size(vector<Size3> *size);
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file);
		void save_param(FILE* file)const;
		void load_param(FILE* file);
		void show(std::ostream &out)const;
		float norm(int num)const;

		float momentum = 0.9f;
		float epsilon = 1e-8f;
		Mat gamma;
		Mat beta;
		Mat	moving_mean;
		Mat moving_var;
	private:	
		bool isVec;
		Mat mean, var;
		vector<Mat> variable[2];
	};
}

