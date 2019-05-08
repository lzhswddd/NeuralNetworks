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
	CONV2D			�����
	POOL			�ػ���
	FULL_CONNECTION ȫ���Ӳ�
	ACTIVATION		�����
	RESHAPE			����ά�Ȳ�
	DROPOUT			���ʹ�ܲ�
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
