#pragma once
#include "layer.h"

namespace nn {
	class pool :
		public Layer
	{
	public:
		pool(string name = "");
		~pool();
		void forword(const Mat &in, Mat &out)const; 
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable);
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const;
		Size3 initialize(Size3 param_size);
		void save(json * jarray, FILE* file)const;
		void load(json & info, FILE* file);
		void show(std::ostream &out)const;

		enum pooltype
		{
			max_pool,
			average_pool
		};
		Size ksize;
		int strides;
		pooltype pool_type;
		vector<Mat> markpoint;

		const Mat MaxPool(const Mat & input, size_t idx);
		const Mat upsample(const Mat & input, size_t idx)const;
		const Mat MaxPool(const Mat & input)const;
		const Mat iMaxPool(const Mat & input, size_t idx)const;
		const Mat AveragePool(const Mat & input)const;
		const Mat iAveragePool(const Mat & input)const;

		static pooltype String2Pooltype(string str);
		static string Pooltype2String(pooltype type);
	};
}