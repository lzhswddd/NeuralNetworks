#ifndef __LAYER_H__
#define __LAYER_H__

#include "Vriable.h"
#include "Function.h"

namespace nn
{
	/**
	@brief ConvInfo ���������
	��Ա
	Size strides ��������
	Point anchor ���ض�Ӧ���������
	anchorĬ��ΪPoint(-1,-1), ���ض�Ӧ���������
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
	@brief Activation �������
	��Ա
	ActivationFunc activation_f �����
	ActivationFunc activation_df ���������
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
