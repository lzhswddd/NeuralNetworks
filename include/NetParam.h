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
	CONV2D			�����
	MAX_POOL		���ֵ�ػ���
	AVERAGE_POOL	ƽ��ֵ�ػ���
	FULL_CONNECTION ȫ���Ӳ�
	ACTIVATION		�����
	RESHAPE			����ά�Ȳ�
	DROPOUT			���ʹ�ܲ�
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
	None				���ṩ�Ż�����
	GradientDescent		�ṩ�ݶ��½���
	Momentum			�ṩ�����ݶ��½���
	NesterovMomentum	�ṩԤ�⶯���ݶ��½���
	Adagrad				�ṩ����Ӧѧϰ���ݶ��½���
	RMSProp				�ṩ��������Ӧѧϰ���ݶ��½���
	Adam				�ṩ����Ӧѧϰ�ʶ����ݶ��½���
	NesterovAdam		�ṩ����Ӧѧϰ��Ԥ�⶯���ݶ��½���
	*/
	enum OptimizerMethod
	{
		None = 0,		//!< ���ṩ�Ż�����
		GradientDescent,//!< �ṩ�ݶ��½���
		Momentum,		//!< �ṩ�����ݶ��½���
		NesterovMomentum,//!< �ṩԤ�⶯���ݶ��½���
		Adagrad,		//!< �ṩ����Ӧѧϰ���ݶ��½���
		RMSProp,		//!< �ṩ��������Ӧѧϰ���ݶ��½���
		Adam,			//!< �ṩ����Ӧѧϰ�ʶ����ݶ��½���
		NesterovAdam	//!< �ṩ����Ӧѧϰ��Ԥ�⶯���ݶ��½���
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
	@brief Activation �������
	��Ա
	ActivationFunc activation_f �����
	ActivationFunc activation_df ���������
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
	@brief Loss ��ʧ������
	��Ա
	LossFunc loss_f ��ʧ����
	LossFunc loss_df ��ʧ��������
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
