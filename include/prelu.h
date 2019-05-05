#pragma once
#include "layer.h"

namespace nn {
	class prelu :
		public parametrics
	{
	public:
		prelu(string name = "");
		~prelu();

		void updateregular();
		void update(const vector<Mat> &d, int *idx);
		void forword(const Mat &in, Mat &out)const;
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable);
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const;
		Size3 initialize(Size3 param_size);
		void append_size(vector<Size3> *size);
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file);
		void save_param(FILE* file)const;
		void load_param(FILE* file);
		void show(std::ostream &out)const;
		float norm(int num)const;

		const Mat PReLU(const Mat & x)const;
		const Mat D_PReLU(const Mat & x)const;

		Mat a;
		vector<Mat> variable;
	};
}

