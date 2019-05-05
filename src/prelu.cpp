#include "prelu.h"
#include "method.h"


nn::prelu::prelu(string name)
	: parametrics(name)
{
	type = PRELU;
}


nn::prelu::~prelu()
{
}

nn::Size3 nn::prelu::initialize(Size3 param_size)
{
	a = zeros(1, 1, param_size.c) + 0.25f;
	if (regularization)
		updateregular();
	return param_size;
}

void nn::prelu::updateregular()
{
	regular = a;
}

void nn::prelu::update(const vector<Mat> &d, int *idx)
{
	a += d[*idx];
	if (regularization)
		updateregular();
	*idx += 1;
}

void nn::prelu::forword(const Mat & in, Mat & out) const
{
	out = PReLU(in);
}

void nn::prelu::forword_train(const vector<Mat> &in, vector<Mat> & out, vector<Mat> & variable)
{
	this->variable = variable;
	for (size_t idx = 0; idx < in.size(); ++idx)
		out[idx] = PReLU(in[idx]);
}

void nn::prelu::back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer, int *number) const
{
	for (size_t idx = 0; idx < in.size(); ++idx) {
		(*dlayer)[dlayer->size() - 1 - *number] += mSum(in[idx], CHANNEL);
		out[idx] = Mult(in[idx], D_PReLU(variable[idx]));
	}
	(*dlayer)[dlayer->size() - 1 - *number] /= (float)in.size(); 
	if (regularization)
		(*dlayer)[dlayer->size() - 1 - *number] += lambda * regular / (float)in.size();
	*number += 1;
}

void nn::prelu::append_size(vector<Size3>* size)
{
	size->push_back(a.size3());
}

void nn::prelu::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;
	jarray->push_back(info);
	method::save_mat(file, a);
}

void nn::prelu::load(json & info, FILE * file)
{
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
	method::load_mat(file, a);
}

void nn::prelu::save_param(FILE * file) const
{
	method::save_mat(file, a);
}

void nn::prelu::load_param(FILE * file)
{
	method::load_mat(file, a);
}

void nn::prelu::show(std::ostream & out) const
{
	out << name << "{" << std::endl;
	out << "Layer-> PReLU" << std::endl;
	out << "ai size-> " << a.size3() << std::endl;
	out << "}";
}

float nn::prelu::norm(int num) const
{
	return a.sum(num, true);
}

const Mat nn::prelu::PReLU(const Mat & x) const
{
	Mat y(x.size3());
	const float *ai = a;
	int c = y.channels();
	float *total_y = y;
	const float *total_x = x;
	for (int j = 0; j < c; ++j) {
		float *p = total_y + j;
		const float *mat = total_x + j;
		for (int i = 0; i < y.rows()*y.cols(); ++i) {
			if (*mat < 0)
				*p = *mat*(*ai);
			else if (*mat > 0)
				*p = *mat;
			else
				*p = 0;
			p += c;
			mat += c;
		}
		ai++;
	}
	return y;
}

const Mat nn::prelu::D_PReLU(const Mat & x) const
{
	Mat y(x.size3());
	const float *ai = a;
	int c = y.channels();
	float *total_y = y;
	const float *total_x = x;
	for (int j = 0; j < c; ++j) {
		float *p = total_y + j;
		const float *mat = total_x + j;
		for (int i = 0; i < y.rows()*y.cols(); ++i) {
			if (*mat < 0)
				*p = *ai;
			else if (*mat > 0)
				*p = 1;
			else
				*p = 0;
			p += c;
			mat += c;
		}
		ai++;
	}
	return y;
}
