#include "Layer.h"
using namespace nn;

LayerInfo nn::Reshape(Size3 size)
{
	LayerInfo info;
	info.type = RESHAPE;
	info.reshapeInfo.size = size;
	return info;
}

LayerInfo nn::Activation(ActivationFunc acti)
{
	LayerInfo info;
	info.type = ACTIVATION;
	info.activation = ActivationInfo(acti);
	return info;
}

LayerInfo nn::Activation(ActivationType acti)
{
	LayerInfo info;
	info.type = ACTIVATION;
	info.activation = ActivationInfo(acti);
	return info;
}

LayerInfo nn::MaxPool(Size poolsize, int strides)
{
	LayerInfo info;
	info.type = MAX_POOL;
	info.pInfo.size = poolsize;
	info.pInfo.strides = strides;
	return info;
}

LayerInfo nn::AveragePool(Size poolsize, int strides)
{
	LayerInfo info;
	info.type = AVERAGE_POOL;
	info.pInfo.size = poolsize;
	info.pInfo.strides = strides;
	return info;
}

LayerInfo nn::Dense(int layer_size, ActivationFunc act_f)
{
	LayerInfo info;
	info.type = FULL_CONNECTION;
	info.fcInfo.size = layer_size;
	info.fcInfo.active = act_f;
	info.convInfo.isact = (act_f != nullptr);
	return info;
}

LayerInfo nn::FullConnect(int layer_size, ActivationFunc act_f)
{
	LayerInfo info;
	info.type = FULL_CONNECTION;
	info.fcInfo.size = layer_size;
	info.fcInfo.active = act_f;
	info.convInfo.isact = (act_f != nullptr);
	return info;
}

LayerInfo nn::Conv2D(ConvInfo cinfo, ActivationFunc act_f)
{
	LayerInfo info;
	info.type = CONV2D;
	info.convInfo = cinfo;
	info.convInfo.active = act_f;
	info.convInfo.isact = (act_f != nullptr);
	return info;
}

LayerInfo nn::Conv2D(int channel, int kern_size, bool is_copy_border, ActivationFunc act_f, Size strides, Point anchor)
{
	LayerInfo info;
	info.type = CONV2D;
	info.convInfo.channel = channel;
	info.convInfo.kern_size = kern_size;
	info.convInfo.is_copy_border = is_copy_border;
	info.convInfo.strides = strides;
	info.convInfo.anchor = anchor;
	info.convInfo.active = act_f;
	info.convInfo.isact = (act_f != nullptr);
	return info;
}

nn::ActivationInfo::ActivationInfo(ActivationType acti)
{
	switch (acti)
	{
	case nn::ACTI_SIGMOID:activation_f = Sigmoid;
		break;
	case nn::ACTI_TANH:activation_f = Tanh;
		break;
	case nn::ACTI_RELU:activation_f = ReLU;
		break;
	case nn::ACTI_ELU:activation_f = ELU;
		break;
	case nn::ACTI_LReLU:activation_f = LReLU;
		break;
	case nn::ACTI_SELU:activation_f = SELU;
		break;
	case nn::ACTI_SOFTMAX:activation_f = Softmax;
		break;
	default:
		break;
	}
}
