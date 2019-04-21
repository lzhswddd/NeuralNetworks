#include "Optimizer.h"
#include <iostream>
#include <fstream>
using namespace nn;
using namespace std;

/*
============================    优化器基类    =========================
*/
Optimizer::Optimizer()
	:step(1e-2f), train(nullptr), method(None) {}
Optimizer::Optimizer(float step) : step(step), train(nullptr), method(None) {}
Optimizer::~Optimizer() {}
void Optimizer::RegisterTrain(Train * train)
{
	this->train = train;
}
bool Optimizer::Enable(const Mat & x, const vector<Mat> & y) const
{
	return !(x.empty() || y.empty() || train == nullptr);
}
void Optimizer::RegisterMethod(OptimizerMethod method)
{
	this->method = method;
}
OptimizerMethod Optimizer::Method() const
{
	return method;
}
/*
========================  注册模型函数(无优化器) ========================
*/
Method::Method() :Optimizer() {}
Method::Method(float step) : Optimizer(step) {}
void Method::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error) {}
Optimizer* Method::minimize()const
{
	Optimizer* train = new Method(*this);
	train->RegisterMethod(None);
	return train;
}
void Method::copyTo(Optimizer * op) const
{
	*((Method*)op) = *this;
}
Mat Method::Params() const
{
	return Mat();
}
/*
=======================    GradientDescent优化器    =======================
*/
GradientDescentOptimizer::GradientDescentOptimizer(float step) : Optimizer(step) {}
void GradientDescentOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	a = a - step * df(a, x)
	*/
	if (!Enable(x.input, x.label))return;
	train->Jacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		dlayer[layer_num] = -step * dlayer[layer_num];
	}
}
Optimizer* GradientDescentOptimizer::minimize()const {
	Optimizer* train = new GradientDescentOptimizer(*this);
	train->RegisterMethod(GradientDescent);
	return train;
}
Mat GradientDescentOptimizer::data(vector<string> &value_name)const
{
	Mat mat(1, 1);
	mat(0) = step;
	value_name.resize(1);
	value_name[0] = "step";
	return mat;
}
void GradientDescentOptimizer::copyTo(Optimizer * op) const
{
	*((GradientDescentOptimizer*)op) = *this;
}
Mat GradientDescentOptimizer::Params() const
{
	return Mat();
}
/*
============================    Momentum优化器    =========================
*/
MomentumOptimizer::MomentumOptimizer(float step) : Optimizer(step) {}
void MomentumOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	ma.resize(size.size());
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		ma[layer_num] = zeros(size[layer_num]);
}
void MomentumOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	ma = momentum*ma + step * df(a, x)
	a = a - ma
	*/
	if (!Enable(x.input, x.label))return;
	train->Jacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		ma[layer_num] = momentum * ma[layer_num] + step * dlayer[layer_num];
		dlayer[layer_num] = -ma[layer_num];
	}
}
Optimizer* MomentumOptimizer::minimize(float momentum)const {
	MomentumOptimizer* train = new MomentumOptimizer(*this);
	train->momentum = momentum;
	train->RegisterMethod(Momentum);
	return (Optimizer*)train;
}
Mat MomentumOptimizer::data(vector<string> &value_name)const
{
	Mat mat(2, 1);
	mat(0) = step;
	mat(1) = momentum;
	vector<string>(2).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "momentum";
	return mat;
}
void MomentumOptimizer::copyTo(Optimizer * op) const
{
	*((MomentumOptimizer*)op) = *this;
}
Mat MomentumOptimizer::Params() const
{
	return (Mat_(1, 1) << momentum);
}
/*
=======================    NesterovMomentum优化器    ======================
*/
NesterovMomentumOptimizer::NesterovMomentumOptimizer(float step) : Optimizer(step){}
void NesterovMomentumOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	ma.resize(size.size());
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		ma[layer_num] = zeros(size[layer_num]);
}
void NesterovMomentumOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	ma = momentum*ma + step * df(a - momentum*ma, x)
	a = a - ma
	*/
	if (!Enable(x.input, x.label))return;
	for (size_t layer_num = 0; layer_num < ma.size(); ++layer_num) {
		dlayer[layer_num] = -momentum * ma[layer_num];
	}
	train->FutureJacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		ma[layer_num] = momentum * ma[layer_num] + step * dlayer[layer_num];
		dlayer[layer_num] = -ma[layer_num];
	}
}
Optimizer* NesterovMomentumOptimizer::minimize(float momentum)const {
	NesterovMomentumOptimizer* train = new NesterovMomentumOptimizer(*this);
	train->momentum = momentum;
	train->RegisterMethod(NesterovMomentum);
	return (Optimizer*)train;
}
Mat NesterovMomentumOptimizer::data(vector<string> &value_name)const
{
	Mat mat(2, 1);
	mat(0) = step;
	mat(1) = momentum;
	vector<string>(2).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "momentum";
	return mat;
}
void NesterovMomentumOptimizer::copyTo(Optimizer * op) const
{
	*((NesterovMomentumOptimizer*)op) = *this;
}
Mat NesterovMomentumOptimizer::Params() const
{
	return (Mat_(1, 1) << momentum);
}
/*
=============================    Adagrad优化器    ============================
*/
AdagradOptimizer::AdagradOptimizer(float step) : Optimizer(step){}
void AdagradOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	alpha.resize(size.size());
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		alpha[layer_num] = zeros(size[layer_num]);
}
void AdagradOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	alpha = alpha + df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	if (!Enable(x.input, x.label))return;
	train->Jacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		alpha[layer_num] = alpha[layer_num] + mPow(dlayer[layer_num], 2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), dlayer[layer_num]);
	}
}
Optimizer * AdagradOptimizer::minimize(float epsilon) const
{
	AdagradOptimizer* train = new AdagradOptimizer(*this);
	train->epsilon = epsilon;
	train->RegisterMethod(Adagrad);
	return (Optimizer *)train;
}
Mat AdagradOptimizer::data(vector<string> &value_name)const
{
	Mat mat(2, 1);
	mat(0) = step;
	mat(1) = epsilon;
	vector<string>(2).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "epsilon";
	return mat;
}
void AdagradOptimizer::copyTo(Optimizer * op) const
{
	*((AdagradOptimizer*)op) = *this;
}
Mat nn::AdagradOptimizer::Params() const
{
	return (Mat_(1, 1) << epsilon);
}
/*
=============================    RMSProp优化器    ============================
*/
RMSPropOptimizer::RMSPropOptimizer(float step) : Optimizer(step) {}
void RMSPropOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	alpha.resize(size.size());
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		alpha[layer_num] = zeros(size[layer_num]);
}
void RMSPropOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	alpha = beta*alpha + (1 - beta)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	if (!Enable(x.input, x.label))return;
	train->Jacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		alpha[layer_num] = decay * alpha[layer_num] + (1 - decay) *  mPow(dlayer[layer_num], 2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), dlayer[layer_num]);
	}
}

Optimizer * RMSPropOptimizer::minimize(float decay, float epsilon) const
{
	RMSPropOptimizer* train = new RMSPropOptimizer(*this);
	train->decay = decay;
	train->epsilon = epsilon;
	train->RegisterMethod(RMSProp);
	return (Optimizer*)train;
}
Mat RMSPropOptimizer::data(vector<string> &value_name)const
{
	Mat mat(3, 1);
	mat(0) = step;
	mat(1) = decay;
	mat(2) = epsilon;
	vector<string>(3).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "decay";
	value_name[2] = "epsilon";
	return mat;
}
void RMSPropOptimizer::copyTo(Optimizer * op) const
{
	*((RMSPropOptimizer*)op) = *this;
}
Mat RMSPropOptimizer::Params() const
{
	return (Mat_(2, 1) << decay, epsilon);
}
/*
=============================    Adam优化器    ============================
*/
AdamOptimizer::AdamOptimizer(float step)
	: Optimizer(step), ma(), alpha() {}
void AdamOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	ma.resize(size.size());
	alpha.resize(size.size());
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num) {
		ma[layer_num] = zeros(size[layer_num]);
		alpha[layer_num] = zeros(size[layer_num]);
	}
}
void AdamOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	ma = beta1*ma + (1 - beta1)*df(a, x)
	alpha = beta2*alpha + (1 - beta2)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	if (!Enable(x.input, x.label))return;
	train->Jacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		Mat d = dlayer[layer_num];
		ma[layer_num] = beta1 * ma[layer_num] + (1 - beta1)*dlayer[layer_num];
		alpha[layer_num] = beta2 * alpha[layer_num] + (1 - beta2)*mPow(dlayer[layer_num], 2);
		//ma[layer_num] /= (1 - beta1);
		//alpha[layer_num] /= (1 - beta2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), ma[layer_num]);
	}
}
Optimizer * AdamOptimizer::minimize(float beta1, float beta2, float epsilon) const
{
	AdamOptimizer* train = new AdamOptimizer(*this);
	train->beta1 = beta1;
	train->beta2 = beta2;
	train->epsilon = epsilon;
	train->RegisterMethod(Adam);
	return (Optimizer*)train;
}
Mat AdamOptimizer::data(vector<string> &value_name)const
{
	Mat mat(4, 1);
	mat(0) = step;
	mat(1) = beta1;
	mat(2) = beta2;
	mat(3) = epsilon;
	value_name.resize(4);
	value_name[0] = "step";
	value_name[1] = "beta1";
	value_name[2] = "beta2";
	value_name[3] = "epsilon";
	return mat;
}
void AdamOptimizer::copyTo(Optimizer * op) const
{
	*((AdamOptimizer*)op) = *this;
}
Mat nn::AdamOptimizer::Params() const
{
	return (Mat_(3, 1) << beta1, beta2, epsilon);
}
/*
=============================    NesterovAdam优化器    ============================
*/
NesterovAdamOptimizer::NesterovAdamOptimizer(float step)
	: Optimizer(step), ma(), alpha() {}
void NesterovAdamOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	ma.resize(size.size());
	alpha.resize(size.size());
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num) {
		ma[layer_num] = zeros(size[layer_num]);
		alpha[layer_num] = zeros(size[layer_num]);
	}
}
void NesterovAdamOptimizer::Run(vector<Mat> &dlayer, const NetData &x, vector<float> &error)
{
	/**
	ma = beta1*ma + (1 - beta1)*df(a - step/sqrt(alpha + epsilon)*ma, x)
	alpha = beta2*alpha + (1 - beta2)*df(a - step/sqrt(alpha + epsilon)*ma, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	if (!Enable(x.input, x.label))return;
	for (size_t layer_num = 0; layer_num < ma.size(); ++layer_num) {
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), ma[layer_num]);
	}
	train->FutureJacobi(x, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		Mat d = dlayer[layer_num];
		ma[layer_num] = beta1 * ma[layer_num] + (1 - beta1)*dlayer[layer_num];
		alpha[layer_num] = beta2 * alpha[layer_num] + (1 - beta2)*mPow(dlayer[layer_num], 2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), ma[layer_num]);
	}
}
Optimizer * NesterovAdamOptimizer::minimize(float beta1, float beta2, float epsilon) const
{
	NesterovAdamOptimizer* train = new NesterovAdamOptimizer(*this);
	train->beta1 = beta1;
	train->beta2 = beta2;
	train->epsilon = epsilon;
	train->RegisterMethod(NesterovAdam);
	return (Optimizer*)train;
}

Mat NesterovAdamOptimizer::data(vector<string> &value_name)const
{
	Mat mat(4, 1);
	mat(0) = step;
	mat(1) = beta1;
	mat(2) = beta2;
	mat(3) = epsilon;
	vector<string>(4).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "beta1";
	value_name[2] = "beta2";
	value_name[3] = "epsilon";
	return mat;
}

void NesterovAdamOptimizer::copyTo(Optimizer * op) const
{
	*((NesterovAdamOptimizer*)op) = *this;
}

Mat NesterovAdamOptimizer::Params() const
{
	return (Mat_(3, 1) << beta1, beta2, epsilon);
}


Optimizer * nn::CreateOptimizer(Optimizer * op)
{
	return CreateOptimizer(op->Method(), op->Step(), op->Params());
}
Optimizer * nn::CreateOptimizer(OptimizerInfo op)
{
	switch (op.type)
	{
	case None:
		return Method().minimize();
	case GradientDescent:
		return GradientDescentOptimizer(op.step).minimize();
	case Momentum:
		return MomentumOptimizer(op.step).minimize(op.momentum);
	case NesterovMomentum:
		return NesterovMomentumOptimizer(op.step).minimize(op.momentum);
	case Adagrad:
		return AdagradOptimizer(op.step).minimize(op.epsilon);
	case RMSProp:
		return RMSPropOptimizer(op.step).minimize(op.decay, op.epsilon);
	case Adam:
		return AdamOptimizer(op.step).minimize(op.beta1, op.beta2, op.epsilon);
	case NesterovAdam:
		return NesterovAdamOptimizer(op.step).minimize(op.beta1, op.beta2, op.epsilon);
	default:
		return nullptr;
	}
}
Optimizer * nn::CreateOptimizer(OptimizerMethod opm, float step, const Mat & value)
{
	switch (opm)
	{
	case None:
		return Method().minimize();
	case GradientDescent:
		return GradientDescentOptimizer(step).minimize();
	case Momentum:
		return MomentumOptimizer(step).minimize(value(0));
	case NesterovMomentum:
		return NesterovMomentumOptimizer(step).minimize(value(0));
	case Adagrad:
		return AdagradOptimizer(step).minimize(value(2));
	case RMSProp:
		return RMSPropOptimizer(step).minimize(value(0), value(2));
	case Adam:
		return AdamOptimizer(step).minimize(value(0), value(1), value(2));
	case NesterovAdam:
		return NesterovAdamOptimizer(step).minimize(value(0), value(1), value(2));
	default:
		return nullptr;
	}
}