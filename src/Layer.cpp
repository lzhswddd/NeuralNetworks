#include "layer.h"
#include "convolution.h"
#include "fullconnection.h"
#include "batchnormalization.h"
#include "pool.h"
#include "activation.h"
#include "costfunction.h"
#include "dropout.h"
#include "reshape.h"
#include "prelu.h"

#include <iostream>
using namespace nn;

Layer * nn::CreateLayer(json & info, FILE* file)
{
	Layer *layer = nullptr;
	switch (Layer::String2Type(info["type"]))
	{
	case nn::NONE:
		layer = (Layer*)new Node(info["name"]);
		break;
	case nn::CONV2D:
		layer = (Layer*)new convolution(info["name"]);
		break;
	case nn::POOL:
		layer = (Layer*)new pool(info["name"]);
		break;
	case nn::FULLCONNECTION:
		layer = (Layer*)new fullconnection(info["name"]);
		break;
	case nn::BATCHNORMALIZATION:
		layer = (Layer*)new batchnormalization(info["name"]);
		break;
	case nn::ACTIVATION:
		layer = (Layer*)new activation(info["name"]);
		break;
	case nn::RESHAPE:
		layer = (Layer*)new reshape(info["name"]);
		break;
	case nn::PRELU:
		layer = (Layer*)new prelu(info["name"]);
		break;
	default:
		break;
	}
	layer->load(info, file);
	return layer;
}

Layer * nn::PReLU(string name)
{
	prelu *info = new prelu(name);
	return (Layer*)info;
}

Layer * nn::BatchNorm(string name)
{
	batchnormalization *info = new batchnormalization(name);
	return (Layer*)info;
}

Layer* nn::Loss(LossFunc loss_f, float weight, string name)
{
	costfunction *info = new costfunction(name);
	info->setfunction(loss_f);
	info->weight = weight;
	return (Layer*)info;
}

Layer* nn::Loss(ReduceType loss_f, float weight, string name)
{
	costfunction *info = new costfunction(name);
	info->setfunction(loss_f);
	info->weight = weight;
	return (Layer*)info;
}

Layer* nn::Loss(LossFunc f, LossFunc df, bool ignore_activate, float weight, string name)
{
	costfunction *info = new costfunction(name);
	info->f = f;
	info->df = df;
	info->ignore_active = ignore_activate;
	info->weight = weight;
	info->weight = weight;
	return (Layer*)info;
}

Layer* nn::Dropout(float dropout_, string name)
{
	dropout *info = new dropout(name);
	info->dropout_probability = dropout_;
	return (Layer*)info;
}

Layer* nn::Reshape(Size3 size, string name)
{
	reshape *info = new reshape(name);
	info->size = size;
	return (Layer*)info;
}

Layer* nn::Activation(ActivationFunc acti, string name)
{
	activation *info = new activation(name);
	info->active = ActivationInfo(acti);
	return (Layer*)info;
}

Layer* nn::Activation(ActivationType acti, string name)
{
	activation *info = new activation(name);
	info->active = ActivationInfo(acti);
	return (Layer*)info;
}

Layer* nn::Activation(ActivationFunc f, ActivationFunc df, string name)
{
	activation *info = new activation(name);
	info->active.f = f;
	info->active.df = df;
	return (Layer*)info;
}

Layer* nn::MaxPool(Size poolsize, int strides, string name)
{
	pool *info = new pool(name);
	info->pool_type = pool::max_pool;
	info->ksize = poolsize;
	info->strides = strides;
	return (Layer*)info;
}

Layer* nn::AveragePool(Size poolsize, int strides, string name)
{
	pool *info = new pool(name);
	info->pool_type = pool::average_pool;
	info->ksize = poolsize;
	info->strides = strides;
	return (Layer*)info;
}

Layer* nn::Dense(int layer_size, ActivationFunc act_f, string name)
{
	return FullConnect(layer_size, act_f, name);
}

Layer* nn::FullConnect(int layer_size, ActivationFunc act_f, string name)
{
	fullconnection *info = new fullconnection(name);
	info->size = layer_size;
	info->active = ActivationInfo(act_f);
	info->isact = (act_f != nullptr);
	return (Layer*)info;
}

Layer* nn::Conv2D(ConvInfo cinfo, ActivationFunc act_f, string name)
{
	convolution *info = new convolution(name);
	info->channel = cinfo.channel;
	info->kern_size = cinfo.kern_size;
	info->strides = cinfo.strides;
	info->anchor = cinfo.anchor;
	info->is_copy_border = cinfo.is_copy_border;
	info->active = ActivationInfo(act_f);
	info->isact = (act_f != nullptr);
	return (Layer*)info;
}

Layer* nn::Conv2D(int channel, int kern_size, bool is_copy_border, ActivationFunc act_f, string name, Size strides, Point anchor)
{
	convolution *info = new convolution(name);
	info->channel = channel;
	info->kern_size = kern_size;
	info->is_copy_border = is_copy_border;
	info->strides = strides;
	info->anchor = anchor;
	info->active = ActivationInfo(act_f);
	info->isact = (act_f != nullptr);
	return (Layer*)info;
}

nn::ActivationInfo::ActivationInfo(ActivationType acti)
{
	switch (acti)
	{
	case nn::ACTI_SIGMOID:
		SetFunc(Sigmoid, &f, &df);
		break;
	case nn::ACTI_TANH:
		SetFunc(Tanh, &f, &df);
		break;
	case nn::ACTI_RELU:
		SetFunc(ReLU, &f, &df);
		break;
	case nn::ACTI_ELU:
		SetFunc(ELU, &f, &df);
		break;
	case nn::ACTI_LReLU:
		SetFunc(LReLU, &f, &df);
		break;
	case nn::ACTI_SELU:
		SetFunc(SELU, &f, &df);
		break;
	case nn::ACTI_SOFTMAX:
		SetFunc(Softmax, &f, &df);
		break;
	default:
		break;
	}
}

nn::ActivationInfo::ActivationInfo(ActivationFunc acti)
{
	SetFunc(acti, &f, &df);
}

string nn::Layer::Type2String(LayerType type)
{
	string type_name = "";
	switch (type)
	{
	case nn::NONE:
		type_name = "NONE";
		break;
	case nn::CONV2D:
		type_name = "CONV2D";
		break;
	case nn::POOL:
		type_name = "POOL";
		break;
	case nn::BATCHNORMALIZATION:
		type_name = "BATCH_NORMALIZATION";
		break;
	case nn::FULLCONNECTION:
		type_name = "FULL_CONNECTION";
		break;
	case nn::ACTIVATION:
		type_name = "ACTIVATION";
		break;
	case nn::RESHAPE:
		type_name = "RESHAPE";
		break;
	case nn::DROPOUT:
		type_name = "DROPOUT";
		break;
	case nn::PRELU:
		type_name = "PRELU";
		break;
	case nn::LOSS:
		type_name = "LOSS";
		break;
	default:
		break;
	}
	return type_name;
}

LayerType nn::Layer::String2Type(string str)
{
	if (str == "NONE")return NONE;
	else if (str == "CONV2D")return CONV2D;
	else if (str == "POOL")return POOL;
	else if (str == "FULL_CONNECTION")return FULLCONNECTION; 
	else if (str == "BATCH_NORMALIZATION")return BATCHNORMALIZATION;
	else if (str == "ACTIVATION")return ACTIVATION;
	else if (str == "RESHAPE")return RESHAPE;
	else if (str == "DROPOUT")return DROPOUT;
	else if (str == "PRELU")return PRELU;
	else if (str == "LOSS")return LOSS;
	else return NONE;
}

std::ostream & nn::operator<<(std::ostream & out, const Layer * layer)
{
	layer->show(out);
	return out;
}
