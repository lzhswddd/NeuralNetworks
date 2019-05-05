#include "reshape.h"



nn::reshape::reshape(string name)
	: Layer(name)
{
	isinit = false;
	isforword = true;
	isback = true;
	isupdate = false;
	isparam = false;
	type = RESHAPE;
}


nn::reshape::~reshape()
{
}

void nn::reshape::forword(const Mat & in, Mat &out) const
{
	out = in;
	out.reshape(size);
}

void nn::reshape::forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> & variable)
{
	front_size = in[0].size3();
	out = in;
	for (size_t idx = 0; idx < in.size(); ++idx) 
		out[idx].reshape(size);
	variable = out;
}

void nn::reshape::back(const vector<Mat> & in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const
{
	out = in; 
	for (size_t idx = 0; idx < in.size(); ++idx)
		out[idx].reshape(front_size);
}

nn::Size3 nn::reshape::initialize(Size3 param_size)
{
	if (param_size.h*param_size.w*param_size.c != size.h*size.w*size.c) {
		fprintf(stderr, "维度不同, 无法改变尺寸!\n");
		throw std::exception("维度不同, 无法改变尺寸!");
	}
	return size;
}

void nn::reshape::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;
	info["size"]["h"] = size.h;
	info["size"]["w"] = size.w;
	info["size"]["c"] = size.c;
	jarray->push_back(info);
}

void nn::reshape::load(json & info, FILE * file)
{
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
	size.h = info["size"]["h"];
	size.w = info["size"]["w"];
	size.c = info["size"]["c"];
}

void nn::reshape::show(std::ostream & out) const
{
	out << name << "{" << std::endl;
	out << "Layer-> Reshape" << std::endl;
	out << "size-> " << size << std::endl;
	out << "}";
}
