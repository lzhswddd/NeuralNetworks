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
		@brief FutureJacobi Ԥ���ſɱȾ���
		ģ�Ͳ����Ƚ��е���������ݶȾ���
		@param input ����
		@param output ���
		@param d_layer ����ݶȾ���
		*/
		void FutureJacobi(
			const NetData &data,
			vector<Mat> & d_layer, vector<float> & error) {}
		/**
		@brief FutureJacobi �ſɱȾ���
		�����ݶȾ���
		@param input ����
		@param output ���
		@param d_layer ����ݶȾ���
		*/
		void Jacobi(
			const NetData &data,
			vector<Mat> & d_layer, vector<float> & error);
		/**
		@brief TrainModel ������ѵ��ģ��
		return ����1��ȷ, 0����, -1�쳣
		@param input ����
		@param label ��ǩ
		@param error ��ʧ�������
		*/
		void Fit(const NetData & data, vector<float> *error = nullptr);
		/**
		@brief TrainModel ������ѵ��ģ��
		return ������ȷ��, -1�쳣
		@param input ����
		@param label ��ǩ
		@param error ��ʧ�������
		*/
		void Fit(const vector<NetData> &data, vector<float> *error = nullptr);

		void Fit(TrainData &trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show = false);
		void Fit(Net *net, TrainData &trainData, OptimizerInfo op_info, int epoch_number, int show_epoch, bool once_show = false);

	protected:
		/**
		@brief BackPropagation ���򴫲�
		@param input ����
		@param output ���
		@param d_layer ����ݶ�ֵ
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