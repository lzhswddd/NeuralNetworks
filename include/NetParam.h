#ifndef __NETPARAM_H__
#define __NETPARAM_H__

#include "Vriable.h"
#include "Function.h"
#include "Mat.h"
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
		NORM_L1 = 0,
		NORM_L2,
		QUADRATIC,
		CROSSENTROPY,
		SOFTMAXCROSSENTROPY,
	};
	/**
	CONV2D			卷积层
	MAX_POOL		最大值池化层
	AVERAGE_POOL	平均值池化层
	FULL_CONNECTION 全连接层
	ACTIVATION		激活层
	RESHAPE			重置维度层
	DROPOUT			随机使能层
	*/
	enum LayerType {
		NONE = 0,
		CONV2D,
		MAX_POOL,
		AVERAGE_POOL,
		FULLCONNECTION,
		ACTIVATION,
		RESHAPE,
		DROPOUT,
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
	template<typename _Tp>
	class NetNode
	{
	public:
		explicit NetNode()
			: data(), child(), parent(nullptr)
		{

		}
		NetNode(const _Tp &data)
			: data(data), child(), parent(nullptr)
		{

		}
		NetNode(const _Tp &data, NetNode<_Tp>* parent)
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
	class FCInfo
	{
	public:
		explicit FCInfo() {}
		bool isact = false;
		int size;
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
	class DropoutInfo
	{
	public:
		explicit DropoutInfo() {}
		float dropout;
	};
	/**
	@brief Loss 损失函数类
	成员
	LossFunc loss_f 损失函数
	LossFunc loss_df 损失函数导数
	*/
	class LossInfo
	{
	public:
		explicit LossInfo() :f(nullptr), df(nullptr) {}
		LossInfo(LossFunc loss_f)
		{
			SetFunc(loss_f, &this->f, &this->df);
			if(loss_f == CrossEntropy || SOFTMAXCROSSENTROPY)
				ignore_acctive = true;
		}
		LossInfo(ReduceType loss_f)
		{
			switch (loss_f)
			{
			case nn::NORM_L1:
				SetFunc(L1, &this->f, &this->df);
				break;
			case nn::NORM_L2:
				SetFunc(L2, &this->f, &this->df);
				break;
			case nn::QUADRATIC:
				SetFunc(Quadratic, &this->f, &this->df);
				break;
			case nn::CROSSENTROPY:
				SetFunc(CrossEntropy, &this->f, &this->df);
				ignore_acctive = true;
				break;
			case nn::SOFTMAXCROSSENTROPY:
				SetFunc(SoftmaxCrossEntropy, &this->f, &this->df);
				ignore_acctive = true;
				break;
			default:
				break;
			}
		}
		void clean()
		{
			f = nullptr;
			df = nullptr;
		}
		LossFunc f;
		LossFunc df;
		bool ignore_acctive = false;
	};
	class NetData
	{
	public:
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
}

#endif
