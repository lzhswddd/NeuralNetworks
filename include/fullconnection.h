#pragma once
#include "layer.h"

namespace nn
{
	class fullconnection :
		public parametrics
	{
	public:
		fullconnection(string name = "");
		~fullconnection();
		void updateregular();
		void update(const vector<Mat> &d, int *idx);
		void forword(const Mat &in, Mat &out)const;
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> & variable);
		void back(const vector<Mat> & in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const;
		Size3 initialize(Size3 param_size);
		void append_size(vector<Size3> *size);
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file);
		void save_param(FILE* file)const;
		void load_param(FILE* file);
		void show(std::ostream &out)const;
		float norm(int num)const;

		bool isact = false;
		int size;
		ActivationInfo active;
		Mat param;
		Mat bias;
	private:
		vector<Mat> variable[2];
	};
}

