#pragma once
#include "layer.h"

namespace nn {
	class convolution :
		public parametrics
	{
	public:
		convolution(string name = "");
		~convolution();
		Size3 initialize(Size3 param_size);
		void updateregular();
		void update(const vector<Mat> &d, int *idx);
		void forword(const Mat &param, Mat &out)const;
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable);
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const;
		void append_size(vector<Size3> *size);
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file);
		void save_param(FILE* file)const;
		void load_param(FILE* file);
		void show(std::ostream &out)const;
		float norm(int num)const;

		bool isact = false;
		int channel;
		int kern_size;
		bool is_copy_border;
		Size strides;
		Point anchor;
		ActivationInfo active;
		Mat param;
		Mat bias;

		const Mat iconv2d(const Mat& input, const Mat& kern, Size3 E_kern_size, bool copyborder = true)const;
		const Mat conv2d(const Mat& input, const Mat& kern)const;
	private:
		vector<Mat> variable[2];
	};
}