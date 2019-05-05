#ifndef __LAYER_H__
#define __LAYER_H__

#include "netParam.h"
#include <fstream>
#include <stdio.h>
#include "json.hpp"
using json = nlohmann::json;

namespace nn
{
	class Layer
	{
	public:
		explicit Layer(string name = "") : name(name), type(NONE), layer_index(0){}
		virtual void forword(const Mat &in, Mat &out)const = 0;
		virtual void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable) = 0;
		virtual void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const = 0;
		virtual Size3 initialize(Size3 param_size) = 0;
		virtual void save(json * jarray, FILE* file)const = 0;
		virtual void load(json & info, FILE* file) = 0;
		virtual void show(std::ostream &out)const = 0;
		string name;
		LayerType type;
		int layer_index;
		bool last = false;
		bool start = false;
		bool isinit;
		bool isforword;
		bool isback;
		bool isupdate;
		bool isparam;

		static string Type2String(LayerType type);
		static LayerType String2Type(string str);
		friend std::ostream & operator << (std::ostream &out, const Layer *layer);
	};
	class parametrics : public Layer
	{
	public:
		explicit parametrics(string name = "") : Layer(name) 
		{
			isinit = true;
			isforword = true;
			isback = true;
			isupdate = true;
			isparam = true;
		}
		virtual void updateregular() = 0;
		virtual void update(const vector<Mat> &d, int *idx) = 0;
		virtual float norm(int num)const = 0;
		virtual void save_param(FILE* file)const = 0;
		virtual void load_param(FILE* file) = 0;
		virtual void append_size(vector<Size3> *size) = 0;
		bool regularization = false;
		float lambda = 0.5f;
		Mat regular;
	};
	class Node : public Layer
	{
	public:
		explicit Node(string name = "") : Layer(name) {
			isinit = false;
			isforword = false;
			isback = false;
			isupdate = false;
			isparam = false;
		}
		void forword(const Mat &in, Mat &out)const { out = in; }
		void forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> &variable) { out = in; }
		void back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer, int *number)const { out = in; }
		Size3 initialize(Size3 size) { return size; }
		void save(json * jarray, FILE* file)const {}
		void load(json & info, FILE* file) {}
		void show(std::ostream &out)const {}
	};

	Layer* CreateLayer(json &info, FILE* file);
	Layer* PReLU(string name = "");
	Layer* BatchNorm(string name = "");
	Layer* Reshape(Size3 size, string name = "");
	Layer* Loss(LossFunc loss_f, float weight = 1, string name = "");
	Layer* Loss(ReduceType loss_f, float weight = 1, string name = "");
	Layer* Loss(LossFunc f, LossFunc df, bool ignore_activate, float weight = 1, string name = "");
	Layer* Dropout(float dropout, string name = "");
	Layer* Activation(ActivationFunc acti, string name = "");
	Layer* Activation(ActivationType acti, string name = "");
	Layer* Activation(ActivationFunc f, ActivationFunc df, string name = "");
	Layer* MaxPool(Size poolsize, int strides, string name = "");
	Layer* AveragePool(Size poolsize, int strides, string name = "");
	Layer* Dense(int layer_size, ActivationFunc act_f = nullptr, string name = "");
	Layer* FullConnect(int layer_size, ActivationFunc act_f = nullptr, string name = "");
	Layer* Conv2D(ConvInfo cinfo, ActivationFunc act_f = nullptr, string name = "");
	Layer* Conv2D(int output_channel, int kern_size, bool is_copy_border = true, ActivationFunc act_f = nullptr, string name = "", Size strides = Size(1, 1), Point anchor = Point(-1, -1));
}

#endif // !__LAYER_H__
