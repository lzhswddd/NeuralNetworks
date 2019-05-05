#include "activation.h"



nn::activation::activation(string name)
	: Layer(name)
{
	isinit = false;
	isforword = true;
	isback = true;
	isupdate = false;
	isparam = false;
	type = ACTIVATION;
}


nn::activation::~activation()
{
}

void nn::activation::forword(const Mat & in, Mat &out) const
{
	out = active.f(in);
}

void nn::activation::forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> & variable)
{
	this->variable = variable;
	for (size_t idx = 0; idx < in.size(); ++idx)
		out[idx] = active.f(in[idx]);
}

void nn::activation::back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer, int *number)const
{
	for (size_t idx = 0; idx < in.size(); ++idx)
		out[idx] = Mult(in[idx], active.df(variable[idx]));
}

nn::Size3 nn::activation::initialize(Size3 param_size)
{
	return param_size;
}

void nn::activation::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;
	info["activate"] = Func2String(active.f);
	jarray->push_back(info);
}

void nn::activation::load(json & info, FILE * file)
{
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
	SetFunc(info["activate"], &active.f, &active.df);
}

void nn::activation::show(std::ostream &out) const
{
	out << name << "{" << std::endl;
	out << "Layer-> Activation" << std::endl;
	out << "activation-> " << Func2String(active.f) << std::endl;
	out << "}";
}
