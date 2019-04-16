#ifndef __FUNCTION_H__
#define __FUNCTION_H__

#include "Mat.h"
#include <string>
using std::string;
namespace nn {
	//LeakyReLU的超参
	const double LReLU_alpha = 0.2;
	//ELU的超参
	const double ELU_alpha = 1.6732632423543772848170429916717;
	//SELU的超参
	const double SELU_scale = 1.0507009873554804934193349852946;

	typedef const Mat(*ActivationFunc)(const Mat &x);
	typedef const Mat(*func)(const Mat & a, const Mat & x);
	typedef const Mat(*d_func)(const Mat & a, const Mat & x, const Mat & dy);
	typedef const Mat(*LossFunc)(const Mat & y, const Mat & y0);
	/**
	@brief softmax函数
	Si = exp(y - max(y)) / sum(exp(y - max(y)))
	*/
	const Mat Softmax(const Mat &y);
	/**
	@brief L1函数
	E = abs(y - y0)
	*/
	const Mat L1(const Mat & y, const Mat & y0);
	/**
	@brief L2函数
	E = (y - y0)^2
	*/
	const Mat L2(const Mat & y, const Mat & y0);
	/**
	@brief quadratic函数
	E = 1/2 * (y - y0)^2
	*/
	const Mat Quadratic(const Mat & y, const Mat & y0);
	/**
	@brief crossentropy函数
	E = -(y * log(y0))
	*/
	const Mat CrossEntropy(const Mat & y, const Mat & y0);
	/**
	@brief sigmoid函数
	y = 1/(1 + exp(-x))
	*/
	const Mat Sigmoid(const Mat & x);
	/**
	@brief tanh函数
	y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
	*/
	const Mat Tanh(const Mat & x);
	/**
	@brief relu函数
	y = {x if x > 0; 0 if x < 0} 
	*/
	const Mat ReLU(const Mat & x);
	/**
	@brief elu函数
	y = {x if x > 0; a*(exp(x) - 1) if x < 0}
	*/
	const Mat ELU(const Mat & x);
	/**
	@brief selu函数
	y = scale*Elu(x)
	*/
	const Mat SELU(const Mat & x);
	/**
	@brief leaky relu函数
	y = {x if x > 0; a*x if x < 0} 
	*/
	const Mat LReLU(const Mat & x);

	/**
	@brief softmax导数
	y = y*(1 - y),
	*/
	const Mat D_Softmax(const Mat & y);
	/**
	@brief quadratic导数
	y = y - y0,
	*/
	const Mat D_Quadratic(const Mat & y, const Mat & y0);
	/**
	@brief crossentropy导数
	y = y - y0,
	*/
	const Mat D_CrossEntropy(const Mat & y, const Mat & y0);
	/**
	@brief sigmoid导数
	y = sigmoid(x) * (1 - sigmoid(x)),
	*/
	const Mat D_Sigmoid(const Mat & x);
	/**
	@brief tanh导数
	y = 4 * d_sigmoid(x),
	*/
	const Mat D_Tanh(const Mat & x);
	/**
	@brief relu导数
	y = {1 if x > 0; 0 if x < 0} 
	*/
	const Mat D_ReLU(const Mat & x);
	/**
	@brief elu导数
	y = {1 if x > 0; a*exp(x) if x < 0}
	*/
	const Mat D_ELU(const Mat & x);
	/**
	@brief selu导数
	y = scale*d_Elu(x)
	*/
	const Mat D_SELU(const Mat & x);
	/**
	@brief leaky relu导数
	y = {1 if x > 0; a if x < 0} 
	*/
	const Mat D_LReLU(const Mat & x);

	//通过函数名注册函数
	void SetFunc(string func_name, LossFunc *f, LossFunc *df);
	//通过函数名注册函数
	void SetFunc(string func_name, ActivationFunc *f, ActivationFunc *df);
	//通过函数指针注册函数
	void SetFunc(LossFunc func, LossFunc *f, LossFunc *df);
	//通过函数指针注册函数
	void SetFunc(ActivationFunc func, ActivationFunc *f, ActivationFunc *df);
	//函数转函数名
	string Func2String(ActivationFunc f);
	//函数转函数名
	string Func2String(LossFunc f);
}
#endif //__FUNCTION_H__
