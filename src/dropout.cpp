#include "dropout.h"


nn::dropout::dropout(string name) :
	Layer(name)
{
	isinit = false;
	isforword = false;
	isback = false;
	isupdate = false;
	isparam = false;
	type = DROPOUT;
}


nn::dropout::~dropout()
{
}


void nn::dropout::forword(const Mat & in, Mat &out) const
{
	out = in;
}

void nn::dropout::forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> & variable)
{
	for (size_t idx = 0; idx < in.size(); ++idx) {
		Mat drop = mThreshold(mRand(0, 1, in[idx].size3(), true), dropout_probability, 0, 1);
		out[idx] = Mult(in[idx], drop);
		out[idx] *= 1.0f / (1 - dropout_probability);
	}
}

void nn::dropout::back(const vector<Mat> & in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const
{
	out = in;
}

nn::Size3 nn::dropout::initialize(Size3 param_size)
{
	return param_size;
}

void nn::dropout::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;
	info["dropout"] = dropout_probability;
	jarray->push_back(info);
}

void nn::dropout::load(json & info, FILE * file)
{
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
	dropout_probability = info["dropout"];
}

void nn::dropout::show(std::ostream & out) const
{
	out << name << "{" << std::endl;
	out << "Layer-> Dropout" << std::endl;
	out << "dropout-> " << dropout_probability << std::endl;
	out << "}";
}
