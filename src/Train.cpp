#include "train.h"
#include "method.h"
#include "optimizer.h"
#include "costfunction.h"
#include "Timer.h"
#include <Windows.h>
using namespace nn;

nn::Train::Train(Net * net)
	: net(net), op(0)
{
}
nn::Train::~Train()
{
	if (op != nullptr) {
		delete op;
		op = nullptr;
	}
}
void nn::Train::Running(bool run)
{
	running = run;
}
size_t nn::Train::outputSize() const
{
	return loss.size();
}
void nn::Train::RegisterNet(Net * net)
{
	this->net = net;
}
void nn::Train::RegisterOptimizer(Optimizer * optimizer)
{
	op = optimizer;
}
void nn::Train::Fit(TrainData::iterator data, vector<float> *error)
{
	if (op == nullptr)return;
	if (data->empty())return;
	if (!net->isEnable()) {
		fprintf(stderr, "error: 网络未初始化!\n");
		throw std::exception("error: 网络未初始化!");
	}
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num) {
		dlayer[layer_num] = zeros(size[layer_num]);
	}
	vector<float> err(loss.size());
	for (float &v : err)
		v = 0;
	BackPropagation(data, dlayer, err);
	int idx = 0;
	Net::update_layer(net->NetTree(), dlayer, idx);
	if (error != nullptr)
		err.swap(*error);
	//}
}
void nn::Train::Fit(TrainOption *option)
{
	Timer t;
	vector<float> error(loss.size());
	for (float &v : error)
		v = 0;
	for (uint epoch = 0; epoch < option->epoch_number; ++epoch) {
		t.Start();
		option->trainData->reset();
		for (int &v : option->trainData->range) {
			if (!running)return;
			TrainData::iterator netData = option->trainData->batches();
			vector<float> err;
			Fit(netData, &err);
			for (size_t i = 0; i < err.size(); ++i)
				error[i] += err[i];
			if (option->everytime_show) {
				if (option->loss_info)
					option->loss_info(pre, 1, err);
				else {
					printf("batch number %d, error ", v + 1);
					for (float &e : err)
						printf("%0.6lf ", e);
					printf("\n");
				}
			}
		}
		if (option->save && epoch % option->save_epoch == option->save_epoch - 1) {
			net->save_param(option->savemodel);
			op->save(option->savemethod);
		}
		if (!option->everytime_show && epoch % option->show_epoch == option->show_epoch - 1) {
			for (float &e : error) {
				e /= (float)option->trainData->len();
			}
			if (option->loss_info)
				option->loss_info(pre, epoch + 1, error);
			else {
				printf("epoch %d, error ", epoch + 1);
				for (float &e : error) {
					printf("%0.6lf ", e);
				}
				printf(" cost time %0.4lf sec\n", t.End() / 1000);
			}
		}
		for (float &e : error)
			e = 0;
	}
	if (option->save) {
		net->save_param(option->savemodel);
		op->save(option->savemethod);
	}
}
void nn::Train::Fit(OptimizerInfo op_info, TrainOption *option)
{
	if (!net) {
		fprintf(stderr, "获取模型失败!\n");
		return;
	}
	OptimizerP optimizer = Optimizer::CreateOptimizer(op_info);
	if (optimizer) {
		if (op != nullptr) {
			delete op;
			op = nullptr;
		}
		op = optimizer;
		initialize();
		op->init(size);
		Fit(option);
		delete op;
		op = nullptr;
	}
	else {
		fprintf(stderr, "获取优化方法失败!\n");
	}
}
void nn::Train::Fit(Optimizer *optimizer, TrainOption *option)
{
	if (!net) {
		fprintf(stderr, "获取模型失败!\n");
		return;
	}
	if (optimizer) {
		if (op != nullptr && op != optimizer) {
			delete op;
			op = nullptr;
		}
		op = optimizer;
		initialize();
		if (option->op_init)
			op->init(size);
		Fit(option);
	}
	else {
		fprintf(stderr, "获取优化方法失败!\n");
	}
}
void nn::Train::Fit(Net * net, Optimizer *op, TrainOption *option)
{
	this->net = net;
	Fit(op, option);
}
void nn::Train::Fit(Net * net, OptimizerInfo op_info, TrainOption *option)
{
	this->net = net;
	option->op_init = true;
	Fit(op_info, option);
}
void nn::Train::BackPropagation(TrainData::iterator data, vector<Mat>& d_layer, vector<float> & error)
{
	if (op != 0)
		op->Run(d_layer, data, error);
}
void nn::Train::FutureJacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float>& error)
{
	int idx = 0;
	Net::update_layer(*net, d_layer, idx);
	vector<Mat> variable(data->size());
	vector<vector<Mat>> transmit(1, vector<Mat>(data->size()));
	for (int idx = 0; idx < data->size(); ++idx) {
		variable[idx] = data->at(idx)->input;
		transmit[0][idx] = data->at(idx)->input;
	}
	Net::forward_train(*net, variable, transmit, 0);
	for (Mat &m : d_layer)
		m.setOpp();
	idx = 0;
	Net::update_layer(*net, d_layer, idx);
	if (loss.empty())
	{
		fprintf(stderr, "没有注册损失函数!");
		throw std::exception("没有注册损失函数!");
	}
	for (size_t i = 0; i < loss.size(); ++i) {
		error[i] = ((const costfunction*)loss[i]->data)->forword(data, &transmit, i);
	}
	JacobiMat(data, d_layer, transmit);
}
void nn::Train::Jacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float> & error)
{
	vector<Mat> variable(data->size());
	vector<vector<Mat>> transmit(1, vector<Mat>(data->size()));
	for (int idx = 0; idx < data->size(); ++idx) {
		variable[idx] = data->at(idx)->input;
		transmit[0][idx] = data->at(idx)->input;
	}
	Net::forward_train(*net, variable, transmit, 0);
	if (loss.empty())
	{
		fprintf(stderr, "没有注册损失函数!");
		throw std::exception("没有注册损失函数!");
	}
	for (size_t i = 0; i < loss.size(); ++i) {
		error[i] = ((const costfunction*)loss[i]->data)->forword(data, &transmit, i);
	}
	JacobiMat(data, d_layer, transmit);
}
void nn::Train::JacobiMat(TrainData::iterator label, vector<Mat> &dlayer, vector<vector<Mat>> &output)
{
	int idx = 0;
	int number = 0;
	for (const NetNode<Layer*> *lossNode : loss) {
		((const costfunction*)lossNode->data)->back(label, &output, idx);
		idx += 1;
	}
	Net::back_train(loss[0]->parent, dlayer, output, number, 0);
}
void nn::Train::initialize()
{
	if (net) {
		loss.clear();
		size.size();
		if (regularization) {
			Net::regularization(*net, lambda);
		}
		Net::initialize_loss(*net, &loss);
		Net::initialize_mat(*net, &size);
		dlayer.resize(size.size());
		reverse(loss.begin(), loss.end());
		op->RegisterTrain(this);
	}
}