#ifndef __NETPARAM_H__
#define __NETPARAM_H__

#include "vriable.h"
#include "function.h"
#include "mat.h"
#include <vector>

using nn::Mat;
using std::vector;

namespace nn
{
	enum ActivationType {
		ACTI_SIGMOID,
		ACTI_TANH,
		ACTI_RELU,
		ACTI_ELU,
		ACTI_LReLU,
		ACTI_SELU,
		ACTI_SOFTMAX
	};
	enum ReduceType {
		NORM_L2,
		QUADRATIC,
		CROSSENTROPY,
		SOFTMAXCROSSENTROPY,
	};
	/**
	CONV2D			卷积层
	POOL			池化层
	FULL_CONNECTION 全连接层
	ACTIVATION		激活层
	RESHAPE			重置维度层
	DROPOUT			随机使能层
	*/
	enum LayerType {
		NONE = 0,
		POOL,
		CONV2D,
		FULLCONNECTION,
		BATCHNORMALIZATION,
		ACTIVATION,
		RESHAPE,
		DROPOUT,
		PRELU,
		LOSS
	};
	/**
	None				不提供优化功能
	GradientDescent		提供梯度下降法
	Momentum			提供动量梯度下降法
	NesterovMomentum	提供预测动量梯度下降法
	Adagrad				提供自适应学习率梯度下降法
	RMSProp				提供改良自适应学习率梯度下降法
	Adam				提供自适应学习率动量梯度下降法
	NesterovAdam		提供自适应学习率预测动量梯度下降法
	*/
	enum OptimizerMethod
	{
		None = 0,		//!< 不提供优化功能
		GradientDescent,//!< 提供梯度下降法
		Momentum,		//!< 提供动量梯度下降法
		NesterovMomentum,//!< 提供预测动量梯度下降法
		Adagrad,		//!< 提供自适应学习率梯度下降法
		RMSProp,		//!< 提供改良自适应学习率梯度下降法
		Adam,			//!< 提供自适应学习率动量梯度下降法
		NesterovAdam	//!< 提供自适应学习率预测动量梯度下降法
	};
	template<class _Tp>
	class NetNode
	{
	public:
		explicit NetNode()
			: data(), child(), parent(nullptr)
		{

		}
		NetNode(_Tp &data)
			: data(data), child(), parent(nullptr)
		{

		}
		NetNode(_Tp &data, NetNode<_Tp>* parent)
			: data(data), child(), parent(parent), sibling()
		{
			
		}
		NetNode(_Tp &&data, NetNode<_Tp>* parent)
			: data(data), child(), parent(parent), sibling()
		{

		}
		NetNode<_Tp>* operator [](int idx)const
		{
			if (idx >= child.size())return nullptr;
			return child[idx];
		}
		_Tp data;
		NetNode<_Tp>* parent;
		vector<NetNode<_Tp>> sibling;
		vector<NetNode<_Tp>> child;
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
		ActivationInfo() : f(nullptr), df(nullptr) {}
		ActivationInfo(ActivationType acti);
		ActivationInfo(ActivationFunc acti);
		ActivationFunc f;
		ActivationFunc df;
	};
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
	};
	class NetData
	{
	public:
		NetData() : input(), label() {}
		NetData(Mat &input, vector<Mat> & label)
			: input(input), label(label) {}
		Mat input;
		vector<Mat> label;
	};

	class OptimizerInfo
	{
	public:
		explicit OptimizerInfo() :type(None) {}
		OptimizerInfo(OptimizerMethod type, float step)
			: type(type), step(step)
		{
			momentum = 0.9f;
			decay = 0.9f;
			epsilon = 1e-7f;
			beta1 = 0.9f;
			beta2 = 0.999f;
		}
		OptimizerMethod type;
		float step;
		float momentum;
		float decay;
		float epsilon;
		float beta1;
		float beta2;
	};

	class Bbox {
	public:
		float score;
		int x1;
		int y1;
		int x2;
		int y2;
		float area;
		float ppoint[10];
		float regreCoord[4];
	};
	inline bool cmpScore(Bbox lsh, Bbox rsh) {
		return lsh.score < rsh.score;
	}
}

#endif
