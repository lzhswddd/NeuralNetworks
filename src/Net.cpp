#include "Net.h"
#include "method.h"
using namespace nn;


Net::Net()
{
}


Net::~Net()
{
}

int nn::Net::initialize(int input_channel)
{
	for (vector<LayerInfo>::iterator layerInfo = config.begin(); layerInfo != config.end(); ++layerInfo) {
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layer.push_back(CreateMat(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*input_channel));
			layer.push_back(CreateMat(1, 1, layerInfo->convInfo.channel));
			input_channel = layerInfo->convInfo.channel;
			if (layerInfo->convInfo.isact) {
				layerInfo = config.insert(layerInfo + 1, Activation(layerInfo->convInfo.active));
			}
			break;
		case nn::FULL_CONNECTION:
			fprintf(stderr, "任意尺寸初始化不允许有全连接层!\n");
			throw FULL_CONNECTION;
		default:
			break;
		}
	}
	return input_channel;
}
Size3 nn::Net::initialize(Size3 input_size)
{
	for (vector<LayerInfo>::iterator layerInfo = config.begin(); layerInfo != config.end(); ++layerInfo) {
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layer.push_back(CreateMat(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*input_size.z));
			layer.push_back(CreateMat(1, 1, layerInfo->convInfo.channel));
			if (!layerInfo->convInfo.is_copy_border)
				input_size = mCalSize(input_size, Size3(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*input_size.z), layerInfo->convInfo.anchor, layerInfo->convInfo.strides);
			else
				input_size.z = layerInfo->convInfo.channel;
			if (layerInfo->convInfo.isact) {
				layerInfo = config.insert(layerInfo + 1, Activation(layerInfo->convInfo.active));
			}
			break;
		case nn::MAX_POOL:
			input_size.x /= layerInfo->pInfo.size.hei;
			input_size.y /= layerInfo->pInfo.size.wid;
			break;
		case nn::AVERAGE_POOL:
			input_size.x /= layerInfo->pInfo.size.hei;
			input_size.y /= layerInfo->pInfo.size.wid;
			break;
		case nn::FULL_CONNECTION:
			if (input_size.z != 1 && input_size.y != 1)
			{
				layerInfo = config.insert(layerInfo, Reshape(Size3(input_size.x*input_size.y*input_size.z, 1, 1)));
				--layerInfo;
			}
			input_size = Size3(input_size.x*input_size.y*input_size.z, 1, 1);
			layer.push_back(CreateMat(layerInfo->fcInfo.size, input_size.x, 1));
			layer.push_back(CreateMat(layerInfo->fcInfo.size, 1, 1));
			if (layerInfo->fcInfo.isact) {
				layerInfo = config.insert(layerInfo + 1, Activation(layerInfo->fcInfo.active));
			}
			break;
		case nn::ACTIVATION:
			break;
		case nn::RESHAPE:
			input_size = layerInfo->reshapeInfo.size;
			break;
		default:
			break;
		}
	}
	return input_size;
}

void nn::Net::clear()
{
	vector<LayerInfo>().swap(config);
	vector<Mat>().swap(layer);
}

Mat nn::Net::forward(const Mat & input) const
{
	if (layer.empty())return Mat();
	if (config.empty())return Mat();
	Mat y = input;
	int layer_num = 0;
	for (vector<LayerInfo>::const_iterator layerInfo = config.begin(); layerInfo != config.end(); ++layerInfo) {
		switch (layerInfo->type)
		{
		case CONV2D:
			y = conv2d(y, layer[layer_num * 2], layerInfo->convInfo.strides, layerInfo->convInfo.anchor, layerInfo->convInfo.is_copy_border) + layer[layer_num * 2 + 1];
			layer_num += 1;
			break;
		case MAX_POOL:
			y = MaxPool(y, layerInfo->pInfo.size, layerInfo->pInfo.strides);
			break;
		case AVERAGE_POOL:
			y = AveragePool(y, layerInfo->pInfo.size, layerInfo->pInfo.strides);
			break;
		case FULL_CONNECTION:
			y = FullConnection(y, layer[layer_num * 2], layer[layer_num * 2 + 1]);
			layer_num += 1;
			break;
		case ACTIVATION:
			y = layerInfo->activation.activation_f(y);
			break;
		case RESHAPE:
			y.reshape(layerInfo->reshapeInfo.size);
		}
	}
	return y;
}

void nn::Net::add(LayerInfo layerInfo)
{
	config.push_back(layerInfo);
}

void nn::Net::addActivation(ActivationFunc act_f)
{
	config.push_back(Activation(act_f));
}

void nn::Net::addMaxPool(Size poolsize, int strides)
{
	config.push_back(MaxPool(poolsize, strides));
}

void nn::Net::addMaxPool(int pool_row, int pool_col, int strides)
{
	config.push_back(MaxPool(Size(pool_row, pool_col), strides));
}

void nn::Net::addAveragePool(Size poolsize, int strides)
{
	config.push_back(AveragePool(poolsize, strides));
}

void nn::Net::addAveragePool(int pool_row, int pool_col, int strides)
{
	config.push_back(AveragePool(Size(pool_row, pool_col), strides));
}

void nn::Net::addDense(int layer_size, ActivationFunc act_f)
{
	config.push_back(Dense(layer_size, act_f));
}

void nn::Net::addFullConnect(int layer_size, ActivationFunc act_f)
{
	config.push_back(FullConnect(layer_size, act_f));
}

void nn::Net::addConv2D(int channel, int kern_size, bool is_copy_border, Size strides, Point anchor, ActivationFunc act_f)
{
	config.push_back(Conv2D(channel, kern_size, is_copy_border, act_f, strides, anchor));
}

const Mat nn::Net::operator()(const Mat & input) const
{
	return forward(input);
}
