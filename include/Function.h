#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include "mat.h"
#include <string>
using std::string;
namespace nn {

	//LeakyReLU�ĳ���
	const float LReLU_alpha = 0.2f;
	//ELU�ĳ���
	const float ELU_alpha = 1.6732632423543772848170429916717f;
	//SELU�ĳ���
	const float SELU_scale = 1.0507009873554804934193349852946f;

	typedef const Mat(*ActivationFunc)(const Mat &x);
	typedef const Mat(*func)(const Mat & a, const Mat & x);
	typedef const Mat(*d_func)(const Mat & a, const Mat & x, const Mat & dy);
	typedef const Mat(*LossFunc)(const Mat & y, const Mat & y0);

	/**
	@brief softmax����
	Si = exp(y - max(y)) / sum(exp(y - max(y)))
	*/
	const Mat Softmax(const Mat &y);
	/**
	@brief L2����
	E = (y - y0)^2
	*/
	const Mat L2(const Mat & y, const Mat & y0);
	/**
	@brief quadratic����
	E = 1/2 * (y - y0)^2
	*/
	const Mat Quadratic(const Mat & y, const Mat & y0);
	/**
	@brief crossentropy����
	E = -(y * log(y0))
	*/
	const Mat CrossEntropy(const Mat & y, const Mat & y0);	
	/**
	@brief softmax + crossentropy����
	E = -(y * log(softmax(y0)))
	*/
	const Mat SoftmaxCrossEntropy(const Mat & y, const Mat & y0);
	/**
	@brief sigmoid����
	y = 1/(1 + exp(-x))
	*/
	const Mat Sigmoid(const Mat & x);
	/**
	@brief tanh����
	y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
	*/
	const Mat Tanh(const Mat & x);
	/**
	@brief relu����
	y = {x if x > 0; 0 if x < 0} 
	*/
	const Mat ReLU(const Mat & x);
	/**
	@brief elu����
	y = {x if x > 0; a*(exp(x) - 1) if x < 0}
	*/
	const Mat ELU(const Mat & x);
	/**
	@brief selu����
	y = scale*Elu(x)
	*/
	const Mat SELU(const Mat & x);
	/**
	@brief leaky relu����
	y = {x if x > 0; a*x if x < 0} 
	*/
	const Mat LReLU(const Mat & x);

	/**
	@brief softmax����
	y = y*(1 - y),
	*/
	const Mat D_Softmax(const Mat & y);
	/**
	@brief L2����
	y = 2 * (y0 - y),
	*/
	const Mat D_L2(const Mat & y, const Mat & y0);
	/**
	@brief quadratic����
	y = y0 - y
	*/
	const Mat D_Quadratic(const Mat & y, const Mat & y0);
	/**
	@brief crossentropy����
	y = y0 - y
	*/
	const Mat D_CrossEntropy(const Mat & y, const Mat & y0);
	/**
	@brief softmax + crossentropy������
	y = y0 - y
	*/
	const Mat D_SoftmaxCrossEntropy(const Mat & y, const Mat & y0);
	/**
	@brief sigmoid����
	y = sigmoid(x) * (1 - sigmoid(x)),
	*/
	const Mat D_Sigmoid(const Mat & x);
	/**
	@brief tanh����
	y = 4 * d_sigmoid(x),
	*/
	const Mat D_Tanh(const Mat & x);
	/**
	@brief relu����
	y = {1 if x > 0; 0 if x < 0} 
	*/
	const Mat D_ReLU(const Mat & x);
	/**
	@brief elu����
	y = {1 if x > 0; a*exp(x) if x < 0}
	*/
	const Mat D_ELU(const Mat & x);
	/**
	@brief selu����
	y = scale*d_Elu(x)
	*/
	const Mat D_SELU(const Mat & x);
	/**
	@brief leaky relu����
	y = {1 if x > 0; a if x < 0} 
	*/
	const Mat D_LReLU(const Mat & x);

	//ͨ��������ע�ắ��
	void SetFunc(string func_name, LossFunc *f, LossFunc *df);
	//ͨ��������ע�ắ��
	void SetFunc(string func_name, ActivationFunc *f, ActivationFunc *df);
	//ͨ������ָ��ע�ắ��
	void SetFunc(LossFunc func, LossFunc *f, LossFunc *df);
	//ͨ������ָ��ע�ắ��
	void SetFunc(ActivationFunc func, ActivationFunc *f, ActivationFunc *df);
	//����ת������
	string Func2String(ActivationFunc f);
	//����ת������
	string Func2String(LossFunc f);
}
#endif //__FUNCTION_H__
