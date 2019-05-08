#include "train.h"
#include "method.h"
#include "optimizer.h"
#include "costfunction.h"
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
	method::update_layer(net->NetTree(), dlayer, idx);
	if (error != nullptr)
		err.swap(*error);
	//}
}

void nn::Train::Fit(TrainData & trainData, int epoch_number, int show_epoch, bool once_show)
{
	LARGE_INTEGER t1, t2, tc;
	QueryPerformanceFrequency(&tc);
	vector<float> error(loss.size()); 
	for (float &v : error)
		v = 0;
	for (int epoch = 0; epoch < epoch_number; ++epoch) {
		QueryPerformanceCounter(&t1);
		trainData.reset();
		for (int &v : trainData.range) {
			TrainData::iterator netData = trainData.batches();
			vector<float> err;
			Fit(netData, &err);
			for (size_t i = 0; i < err.size(); ++i)
				error[i] += err[i];
			if (once_show && epoch == 0) {
				printf("batch number %d, error ", v + 1);
				for (float &e : err)
					printf("%0.6lf ", e);
				printf("\n");
			}
		}
		if (epoch % show_epoch == show_epoch - 1) {
			QueryPerformanceCounter(&t2);
			printf("epoch %d, error ", epoch + 1);
			for (float &e : error) {
				printf("%0.6lf ", e / (float)trainData.len());
			}
			printf(" cost time %0.4lf sec\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
		}
		for (float &e : error)
			e = 0;
	}
}

void nn::Train::Fit(TrainData & trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show)
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
		Fit(trainData, epoch_number, show_epoch, once_show);
		delete op;
		op = nullptr;
	}
	else {
		fprintf(stderr, "获取优化方法失败!\n");
	}
}

void nn::Train::Fit(TrainData & trainData, Optimizer *optimizer, int epoch_number, int show_epoch, bool once_show, bool op_init)
{
	if (!net) {
		fprintf(stderr, "获取模型失败!\n");
		return;
	}
	if (optimizer) {
		if (op != nullptr) {
			delete op;
			op = nullptr;
		}
		op = optimizer;
		initialize();
		if (op_init)
			op->init(size);
		Fit(trainData, epoch_number, show_epoch, once_show);
	}
	else {
		fprintf(stderr, "获取优化方法失败!\n");
	}
}

void nn::Train::Fit(Net * net, TrainData & trainData, Optimizer *op, int epoch_number, int show_epoch, bool once_show, bool op_init)
{
	this->net = net;
	Fit(trainData, op, epoch_number, show_epoch, once_show, op_init);
}

void nn::Train::Fit(Net * net, TrainData & trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show)
{
	this->net = net;
	Fit(trainData, op_info, epoch_number, show_epoch, once_show);
}

void nn::Train::BackPropagation(TrainData::iterator data, vector<Mat>& d_layer, vector<float> & error)
{
	if (op != 0)
		op->Run(d_layer, data, error);
}

void nn::Train::FutureJacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float>& error)
{
	int idx = 0;
	method::update_layer(*net, d_layer, idx);
	vector<Mat> variable(data->size());
	vector<vector<Mat>> transmit(1, vector<Mat>(data->size()));
	for (int idx = 0; idx < data->size(); ++idx) {
		variable[idx] = data->at(idx)->input;
		transmit[0][idx] = data->at(idx)->input;
	}
	method::forward_train(*net, variable, transmit, 0);
	for (Mat &m : d_layer)
		m.setOpp();
	idx = 0;
	method::update_layer(*net, d_layer, idx);
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
	method::forward_train(*net, variable, transmit, 0);
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
	method::back_train(loss[0]->parent, dlayer, output, number, 0);
}

void nn::Train::initialize()
{
	if (net) {
		loss.clear();
		size.size();
		if (regularization) {
			method::regularization(*net, lambda);
		}
		method::initialize_loss(*net, &loss);
		method::initialize_mat(*net, &size);
		dlayer.resize(size.size());
		reverse(loss.begin(), loss.end());
		op->RegisterTrain(this);
	}
}
