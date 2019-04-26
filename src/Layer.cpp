#include "Layer.h"
#include <iostream>
using namespace nn;

Layer nn::Loss(LossFunc loss_f, float weight, string name)
{
	Layer info(name);
	info.type = LOSS;
	info.loss = LossInfo(loss_f);
	info.loss.weight = weight;
	return info;
}

Layer nn::Loss(ReduceType loss_f, float weight, string name)
{
	Layer info(name);
	info.type = LOSS;
	info.loss = LossInfo(loss_f);
	info.loss.weight = weight;
	return info;
}

Layer nn::Loss(LossFunc f, LossFunc df, bool ignore_activate, float weight, string name)
{
	Layer info(name);
	info.type = LOSS;
	info.loss.f = f;
	info.loss.df = df;
	info.loss.ignore_active = ignore_activate;
	info.loss.weight = weight;
	return info;
}

Layer nn::Dropout(float dropout, string name)
{
	Layer info(name);
	info.type = DROPOUT;
	info.dropoutInfo.dropout = dropout;
	return info;
}

std::ostream & nn::operator<<(std::ostream & out, const Layer & layer)
{
	switch (layer.type)
	{
	case CONV2D:
		out << layer.name << "{" << std::endl;
		out << "Layer-> Conv2D" << std::endl;
		out << "kern size-> " << layer.param.size3() << std::endl;
		out << "bias size-> " << layer.bias.size3() << std::endl;
		out << "strides-> " << layer.convInfo.strides << std::endl;
		out << "anchor-> " << layer.convInfo.anchor << std::endl;
		out << "is_copy_border-> " << (layer.convInfo.is_copy_border ? "true" : "false") << std::endl;
		if (layer.convInfo.isact)
			out << "activation-> " << Func2String(layer.active.f) << std::endl;
		out << "}";
		break;
	case MAX_POOL:
		out << layer.name << "{" << std::endl;
		out << "Layer-> MaxPool" << std::endl;
		out << "size-> " << layer.pInfo.size << std::endl;
		out << "strides-> " << layer.pInfo.strides << std::endl;
		out << "}";
		break;
	case AVERAGE_POOL:
		out << layer.name << "{" << std::endl;
		out << "Layer-> AveragePool" << std::endl;
		out << "size-> " << layer.pInfo.size << std::endl;
		out << "strides-> " << layer.pInfo.strides << std::endl;
		out << "}";
		break;
	case FULLCONNECTION:
		out << layer.name << "{" << std::endl;
		out << "Layer-> FullConnection" << std::endl;
		out << "param size-> " << layer.param.mSize() << std::endl;
		out << "bias size-> " << layer.bias.mSize() << std::endl;
		if (layer.fcInfo.isact)
			out << "activation-> " << Func2String(layer.active.f) << std::endl;
		out << "}";
		break;
	case ACTIVATION:
		out << layer.name << "{" << std::endl;
		out << "Layer-> Activation" << std::endl;
		out << "activation-> " << Func2String(layer.active.f) << std::endl;
		out << "}";
		break;
	case RESHAPE:
		break;
	case DROPOUT:
		out << layer.name << "{" << std::endl;
		out << "Layer-> Dropout" << std::endl;
		out << "dropout-> " << layer.dropoutInfo.dropout << std::endl;
		out << "}";
	case LOSS:
		out << layer.name << "{" << std::endl;
		out << "Layer-> Loss" << std::endl;
		out << "loss-> " << Func2String(layer.loss.f) << std::endl;
		out << "ignore_active-> " << (layer.loss.ignore_active ? "true" : "false") << std::endl;
		out << "}";
	default:
		break;
	}
	return out;
}

Layer nn::Reshape(Size3 size, string name)
{
	Layer info(name);
	info.type = RESHAPE;
	info.reshapeInfo.size = size;
	return info;
}

Layer nn::Activation(ActivationFunc acti, string name)
{
	Layer info(name);
	info.type = ACTIVATION;
	info.active = ActivationInfo(acti);
	return info;
}

Layer nn::Activation(ActivationType acti, string name)
{
	Layer info(name);
	info.type = ACTIVATION;
	info.active = ActivationInfo(acti);
	return info;
}

Layer nn::Activation(ActivationFunc f, ActivationFunc df, string name)
{
	Layer info(name);
	info.type = ACTIVATION;
	info.active.f = f;
	info.active.df = df;
	return info;
}

Layer nn::MaxPool(Size poolsize, int strides, string name)
{
	Layer info(name);
	info.type = MAX_POOL;
	info.pInfo.size = poolsize;
	info.pInfo.strides = strides;
	return info;
}

Layer nn::AveragePool(Size poolsize, int strides, string name)
{
	Layer info(name);
	info.type = AVERAGE_POOL;
	info.pInfo.size = poolsize;
	info.pInfo.strides = strides;
	return info;
}

Layer nn::Dense(int layer_size, ActivationFunc act_f, string name)
{
	Layer info(name);
	info.type = FULLCONNECTION;
	info.fcInfo.size = layer_size;
	info.active = ActivationInfo(act_f);
	info.fcInfo.isact = (act_f != nullptr);
	return info;
}

Layer nn::FullConnect(int layer_size, ActivationFunc act_f, string name)
{
	Layer info(name);
	info.type = FULLCONNECTION;
	info.fcInfo.size = layer_size;
	info.active = ActivationInfo(act_f);
	info.fcInfo.isact = (act_f != nullptr);
	return info;
}

Layer nn::Conv2D(ConvInfo cinfo, ActivationFunc act_f, string name)
{
	Layer info(name);
	info.type = CONV2D;
	info.convInfo = cinfo;
	info.active = ActivationInfo(act_f);
	info.convInfo.isact = (act_f != nullptr);
	return info;
}

Layer nn::Conv2D(int channel, int kern_size, bool is_copy_border, ActivationFunc act_f, string name, Size strides, Point anchor)
{
	Layer info(name);
	info.type = CONV2D;
	info.convInfo.channel = channel;
	info.convInfo.kern_size = kern_size;
	info.convInfo.is_copy_border = is_copy_border;
	info.convInfo.strides = strides;
	info.convInfo.anchor = anchor;
	info.active = ActivationInfo(act_f);
	info.convInfo.isact = (act_f != nullptr);
	return info;
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
	case nn::MAX_POOL:
		type_name = "MAX_POOL";
		break;
	case nn::AVERAGE_POOL:
		type_name = "AVERAGE_POOL";
		break;
	case nn::FULLCONNECTION:
		type_name = "FULLCONNECTION";
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
	else if (str == "MAX_POOL")return MAX_POOL;
	else if (str == "AVERAGE_POOL")return AVERAGE_POOL;
	else if (str == "FULLCONNECTION")return FULLCONNECTION;
	else if (str == "ACTIVATION")return ACTIVATION;
	else if (str == "RESHAPE")return RESHAPE;
	else if (str == "DROPOUT")return DROPOUT;
	else if (str == "LOSS")return LOSS;
	else return NONE;
}
