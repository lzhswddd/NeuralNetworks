#include "fullconnection.h"
#include "method.h"


nn::fullconnection::fullconnection(string name) :
	parametrics(name)
{
	type = FULLCONNECTION;
}


nn::fullconnection::~fullconnection()
{
}

void nn::fullconnection::updateregular()
{
	regular = param;
}

void nn::fullconnection::update(const vector<Mat> &d, int *idx)
{
	param += d[(*idx)++];
	bias += d[(*idx)++];
	if (regularization)
		updateregular();
}

nn::Size3 nn::fullconnection::initialize(Size3 param_size)
{
	param = method::Xavier(size, param_size.h, 1, param_size.h, size);
	bias = zeros(size, 1); 
	if (regularization)
		updateregular();
	return Size3(size, 1, 1);
}

void nn::fullconnection::forword(const Mat & in, Mat & out) const
{
	out = param * in + bias;
	if (isact)
		out = active.f(out);
}

void nn::fullconnection::forword_train(const vector<Mat> & in, vector<Mat> & out, vector<Mat> & variable)
{
	this->variable[0] = variable;
	for (size_t idx = 0; idx < in.size(); ++idx)
		out[idx] = param * in[idx] + bias;
	variable = out;
	this->variable[1] = variable;
	if (isact)
		for (size_t idx = 0; idx < in.size(); ++idx)
			out[idx] = active.f(out[idx]);
}

void nn::fullconnection::back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer, int *number)const
{
	for (size_t idx = 0; idx < in.size(); ++idx) {
		if (!last && isact)
			out[idx] = Mult(in[idx], active.df(variable[1][idx]));
		else
			out[idx] = in[idx];
		(*dlayer)[dlayer->size() - 1 - *number] += out[idx];
		(*dlayer)[dlayer->size() - 2 - *number] += out[idx] * variable[0][idx].t();
		if (!start)
			out[idx] = param.t() * out[idx];
	}
	(*dlayer)[dlayer->size() - 1 - *number] /= (float)in.size();
	(*dlayer)[dlayer->size() - 2 - *number] /= (float)in.size(); 
	if (regularization)
		(*dlayer)[dlayer->size() - 2 - *number] += lambda * regular;
	*number += 2;
}

void nn::fullconnection::append_size(vector<Size3>* size)
{
	size->push_back(param.size3());
	size->push_back(bias.size3());
}

void nn::fullconnection::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;
	info["isact"] = isact;
	if (isact)
		info["activate"] = Func2String(active.f);
	info["size"] = size;
	jarray->push_back(info);
	save_param(file);
}

void nn::fullconnection::load(json & info, FILE * file)
{
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
	isact = info["isact"];
	size = info["size"];
	if (isact)
		SetFunc(info["activate"], &active.f, &active.df);
	load_param(file);
}

void nn::fullconnection::save_param(FILE * file) const
{
	method::save_mat(file, param);
	method::save_mat(file, bias);
}

void nn::fullconnection::load_param(FILE * file)
{
	method::load_mat(file, param);
	method::load_mat(file, bias);
}

void nn::fullconnection::show(std::ostream & out) const
{
	out << name << "{" << std::endl;
	out << "Layer-> FullConnection" << std::endl;
	out << "param size-> " << param.mSize() << std::endl;
	out << "bias size-> " << bias.mSize() << std::endl;
	if (isact)
		out << "activation-> " << Func2String(active.f) << std::endl;
	out << "}";
}

float nn::fullconnection::norm(int num) const
{
	return param.sum(num, true);
}
