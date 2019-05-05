#include "batchnormalization.h"
#include "method.h"


nn::batchnormalization::batchnormalization(string name)
	: parametrics(name)
{
	type = BATCHNORMALIZATION;
}


nn::batchnormalization::~batchnormalization()
{
}

nn::Size3 nn::batchnormalization::initialize(Size3 param_size)
{
	Size3 size(1, 1, 1);
	if (param_size.c == 1) {
		size.h = param_size.h;
	}
	else {
		size.c = param_size.c;
	}
	gamma = ones(size);
	beta = zeros(size);
	moving_mean = zeros(size);
	moving_var = zeros(size);
	return param_size;
}

void nn::batchnormalization::updateregular()
{
	regular = gamma;
}

void nn::batchnormalization::update(const vector<Mat>& d, int * idx)
{
	gamma += d[(*idx)++];
	beta += d[(*idx)++];
	if (regularization)
		updateregular();
}

void nn::batchnormalization::forword(const Mat & in, Mat & out) const
{
	out = (in - moving_mean) / (moving_var + epsilon).sqrt();
}

void nn::batchnormalization::forword_train(const vector<Mat>& in, vector<Mat>& out, vector<Mat>& variable)
{
	this->variable[0] = in;
	float size = (float)in[0].size3().area();
	mean = zeros(gamma.size3());
	var = zeros(gamma.size3());
	vector<Mat> temp(in.size());
	vector<Mat>::iterator iter = temp.begin();
	for (const Mat &m : in) {
		*iter = mSum(m, CHANNEL);
		mean += *iter;
		iter += 1;
	}
	mean /= (float)in.size()*size;
	for (const Mat &m : temp)
		var += (m - mean).pow(2);
	var /= (float)in.size()*size;
	moving_mean = momentum * moving_mean + (1.0f - momentum) * mean;
	moving_var = momentum * moving_var + (1.0f - momentum) * var;
	for (size_t idx = 0; idx < in.size(); ++idx) {
		out[idx] = (in[idx] - mean) / (var + epsilon).sqrt();
	}
	this->variable[1] = out;
	for (size_t idx = 0; idx < in.size(); ++idx) {
		out[idx] = Mult(out[idx], gamma) + beta;
		//out[idx].save("test.txt", false);
	}
	variable = out;
}

void nn::batchnormalization::back(const vector<Mat>& in, vector<Mat>& out, vector<Mat>* dlayer, int * number) const
{
	float N = (float)in.size();
	Size3 size = in[0].size3();
	vector<Mat> d(in.size());
	vector<Mat> dx(in.size());
	Mat dv = zeros(in[0].size3());
	Mat dm = zeros(in[0].size3());
	Mat err = var + epsilon;
	for (size_t idx = 0; idx < in.size(); ++idx) {
		(*dlayer)[dlayer->size() - 1 - *number] += mSum(in[idx], CHANNEL);
		(*dlayer)[dlayer->size() - 2 - *number] += mSum(Mult(in[idx], variable[1][idx]), CHANNEL);
		dx[idx] = Mult(in[idx], gamma);
		d[idx] = variable[0][idx] - mean;
		dv += Mult(Mult(dx[idx], d[idx]), -0.5f / err.pow(1.5f));
		dm += Mult(dx[idx], -1.0f / err.sqrt());
	}
	//(*dlayer)[dlayer->size() - 1 - *number] /= N;
	//(*dlayer)[dlayer->size() - 2 - *number] /= N;
	for (size_t idx = 0; idx < in.size(); ++idx) {
		out[idx] = Mult(dx[idx], 1.0f / err.sqrt()) + Mult(dv, (2.0f*(d[idx])) / N) + dm * 1.0f / N;
		//out[idx].save("test.txt", false);
	}
	*number += 2;
}

void nn::batchnormalization::append_size(vector<Size3>* size)
{
	size->push_back(gamma.size3());
	size->push_back(beta.size3());
}

void nn::batchnormalization::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;;
	jarray->push_back(info);
	save_param(file);
}

void nn::batchnormalization::load(json & info, FILE * file)
{
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
	load_param(file);
}

void nn::batchnormalization::save_param(FILE * file) const
{
	method::save_mat(file, moving_mean);
	method::save_mat(file, moving_var);
}

void nn::batchnormalization::load_param(FILE * file)
{
	method::load_mat(file, moving_mean);
	method::load_mat(file, moving_var);
}

void nn::batchnormalization::show(std::ostream & out) const
{
	out << name << "{" << std::endl;
	out << "Layer-> batchnormalization" << std::endl;
	out << "size-> " << gamma.size3() << std::endl;
	out << "}";
}

float nn::batchnormalization::norm(int num) const
{
	return gamma.sum(num, true) + beta.sum(num, true);
}
