#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "net.h"
#include "netparam.h"
#include <vector>
#include <string>
using std::vector;
using std::string;

namespace nn {
	class TrainOption
	{
	public:
		typedef void(*LOSS_INFO)(void*, int, vector<float>);
		TrainData *trainData;
		uint epoch_number;
		int show_epoch;
		bool op_init = true;
		bool everytime_show = false;
		LOSS_INFO loss_info = nullptr;
		bool save = false;
		int save_epoch;
		string savemodel;
		string savemethod;

		void *pre = nullptr;
	};

	class Optimizer;
	class Train
	{
	public:
		Train(Net *net = nullptr);
		~Train();
		void Running(bool run);
		size_t outputSize()const;
		void RegisterNet(Net *net);
		void RegisterOptimizer(Optimizer *optimizer);
		/**
		@brief FutureJacobi 预测雅可比矩阵
		模型参数先进行迭代后计算梯度矩阵
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度矩阵
		*/
		void FutureJacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float> &error);
		/**
		@brief FutureJacobi 雅可比矩阵
		计算梯度矩阵
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度矩阵
		*/
		void Jacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float> &error);
		/**
		@brief TrainModel 多批次训练模型
		@param input 输入
		@param label 标签
		@param error 损失函数输出
		*/
		void Fit(TrainData::iterator data, vector<float> *error = nullptr);

		void Fit(TrainOption *option);
		void Fit(OptimizerInfo op_info, TrainOption *option);
		void Fit(Optimizer *op, TrainOption *option);
		void Fit(Net *net, Optimizer *op, TrainOption *option);
		void Fit(Net *net, OptimizerInfo op_info, TrainOption *option);
		void initialize();

		float lambda = 1.0f;
		bool regularization = false;
		void *pre = nullptr;
		volatile bool running = true;
	protected:
		/**
		@brief BackPropagation 反向传播
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度值
		*/
		void BackPropagation(TrainData::iterator data, vector<Mat> & d_layer, vector<float> & error);
		void JacobiMat(TrainData::iterator label, vector<Mat> &d_layer, vector<vector<Mat>> &output);

	private:
		Net *net;
		Optimizer* op;
		vector<Mat> dlayer;
		vector<Size3> size;
		vector<NetNode<Layer*>*> loss;
	};
}

#endif // __TRAIN_H__