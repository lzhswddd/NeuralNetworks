#pragma once
#include "layer.h"

namespace nn {
	class activation :
		public Layer
	{
	public:
		activation(string name = "");
		~activation();

		void forword(const Mat &in, Mat &out)const;
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable);
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const;
		Size3 initialize(Size3 param_size);
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file); 
		void show(std::ostream &out)const;

		vector<Mat> variable;
		ActivationInfo active;
	};
}

