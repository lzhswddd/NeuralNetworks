#ifndef __LAYER_H__
#define __LAYER_H__

#include "Vriable.h"
#include "Function.h"

namespace nn
{
	/**
	@brief ConvInfo 卷积参数类
	成员
	Size strides 滑动步长
	Point anchor 像素对应卷积核坐标
	anchor默认为Point(-1,-1), 像素对应卷积核中心
	*/
	class ConvInfo
	{
	public:
		explicit ConvInfo() {}
		bool isact = false;
		int channel;
		int kern_size;
		Size strides;
		Point anchor;
		bool is_copy_border;
		ActivationFunc active;
	};
	/**
	@brief Activation 激活函数类
	成员
	ActivationFunc activation_f 激活函数
	ActivationFunc activation_df 激活函数导数
	*/
	class ActivationInfo
	{
	public:
		ActivationInfo() :activation_f(nullptr) {}
		ActivationInfo(ActivationType acti);
		ActivationInfo(ActivationFunc acti) :activation_f(acti) {}
		ActivationFunc activation_f;
	};
	class FCInfo
	{
	public:
		explicit FCInfo() {}
		bool isact = false;
		int size;
		ActivationFunc active;
	};
	class PoolInfo
	{
	public:
		explicit PoolInfo() {}
		Size size;
		int strides;
	}; 
	class ReShapeInfo
	{
	public:
		explicit ReShapeInfo() {}
		Size3 size;
	};
	class LayerInfo
	{
	public:
		explicit LayerInfo() {}
		LayerType type;
		PoolInfo pInfo;
		FCInfo fcInfo; 
		ConvInfo convInfo;
		ReShapeInfo reshapeInfo;
		ActivationInfo activation;
	};
	LayerInfo Reshape(Size3 size);
	LayerInfo Activation(ActivationFunc acti);
	LayerInfo Activation(ActivationType acti);
	LayerInfo MaxPool(Size poolsize, int strides);
	LayerInfo AveragePool(Size poolsize, int strides);
	LayerInfo Dense(int layer_size, ActivationFunc act_f = nullptr);
	LayerInfo FullConnect(int layer_size, ActivationFunc act_f = nullptr);
	LayerInfo Conv2D(ConvInfo cinfo, ActivationFunc act_f = nullptr);
	LayerInfo Conv2D(int output_channel, int kern_size, bool is_copy_border = true, ActivationFunc act_f = nullptr, Size strides = Size(1, 1), Point anchor = Point(-1, -1));
}

#endif // !__LAYER_H__
