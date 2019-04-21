#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "Net.h"
#include "NetParam.h"
#include <vector>
#include <string>
using std::vector;
using std::string;

namespace nn {
	class Optimizer;
	class Train
	{
	public:
		Train(Net *net = nullptr);
		~Train();
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
		void FutureJacobi(
			const NetData &data,
			vector<Mat> & d_layer, vector<float> & error) {}
		/**
		@brief FutureJacobi 雅可比矩阵
		计算梯度矩阵
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度矩阵
		*/
		void Jacobi(
			const NetData &data,
			vector<Mat> & d_layer, vector<float> & error);
		/**
		@brief TrainModel 单批次训练模型
		return 返回1正确, 0错误, -1异常
		@param input 输入
		@param label 标签
		@param error 损失函数输出
		*/
		void Fit(const NetData & data, vector<float> *error = nullptr);
		/**
		@brief TrainModel 多批次训练模型
		return 返回正确率, -1异常
		@param input 输入
		@param label 标签
		@param error 损失函数输出
		*/
		void Fit(const vector<NetData> &data, vector<float> *error = nullptr);

		void Fit(TrainData &trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show = false);
		void Fit(Net *net, TrainData &trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show = false);

	protected:
		/**
		@brief BackPropagation 反向传播
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度值
		*/
		void BackPropagation(const NetData &data, vector<Mat> & d_layer, vector<float> & error);
		void JacobiMat(const vector<Mat> &label, vector<Mat> &d_layer, vector<vector<Mat>> &variable, vector<Mat> &output);
		void initialize();
	private:
		Net *net;
		Optimizer* op;
		vector<Mat> dlayer;
		vector<Size3> size;
		vector<NetNode<Layer>*> loss;
	};
}

#endif // __TRAIN_H__