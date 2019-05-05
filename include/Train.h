#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "net.h"
#include "netparam.h"
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
		@brief FutureJacobi Ԥ���ſɱȾ���
		ģ�Ͳ����Ƚ��е���������ݶȾ���
		@param input ����
		@param output ���
		@param d_layer ����ݶȾ���
		*/
		void FutureJacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float> &error) {}
		/**
		@brief FutureJacobi �ſɱȾ���
		�����ݶȾ���
		@param input ����
		@param output ���
		@param d_layer ����ݶȾ���
		*/
		void Jacobi(TrainData::iterator data, vector<Mat>& d_layer, vector<float> &error);
		/**
		@brief TrainModel ������ѵ��ģ��
		@param input ����
		@param label ��ǩ
		@param error ��ʧ�������
		*/
		void Fit(TrainData::iterator data, vector<float> *error = nullptr);

		void Fit(TrainData &trainData, int epoch_number, int show_epoch, bool once_show = false);
		void Fit(TrainData &trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show = false);
		void Fit(TrainData &trainData, Optimizer *op, int epoch_number, int show_epoch, bool once_show = false, bool op_init = false);
		void Fit(Net *net, TrainData &trainData, Optimizer *op, int epoch_number, int show_epoch, bool once_show = false, bool op_init = false);
		void Fit(Net *net, TrainData &trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show = false);
		void initialize();

		float lambda = 1.0f;
		bool regularization = false;
	protected:
		/**
		@brief BackPropagation ���򴫲�
		@param input ����
		@param output ���
		@param d_layer ����ݶ�ֵ
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