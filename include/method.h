#ifndef __METHOD_H__
#define __METHOD_H__

#include "layer.h"
#include "vriable.h"
#include <fstream>
#include "json.hpp"
using json = nlohmann::json;

using nn::Mat;
using nn::Size3;
using nn::Size;
using nn::Point;
using nn::ConvInfo;

namespace nn
{
	class method
	{
	public:
		static const Mat Xavier(int row, int col, int channel);
		/**
		@brief CreateMat �����������
		��СΪrow*col*channel, Ԫ�ط�Χ[low, top]
		@param row ��������
		@param col ��������
		@param channel ����ͨ����
		@param low ����
		@param top ����
		*/
		static const Mat Random(int row, int col, int channel = 1);
		/**
		@brief CreateMat �����������
		��СΪrow*col*(channel_output - channel_input), Ԫ�ط�Χ[low, top]
		@param row ��������
		@param col ��������
		@param channel_input �������ͨ����
		@param channel_output �������ͨ����
		@param low ����
		@param top ����
		*/
		static const Mat Random(int row, int col, int channel_input, int channel_output);

		static NetNode<Layer*> CreateNode(Layer *data, NetNode<Layer*>* parent);
		static void save_layer(const NetNode<Layer*>* tree, json *j, FILE *data);
		static void save_param_layer(const NetNode<Layer*>* tree, FILE *data);
		static void load_layer(NetNode<Layer*>* tree, FILE *data);
		static int insert_layer(NetNode<Layer*>* tree, string name, Layer *layer, bool sibling);
		static void update_layer(NetNode<Layer*>* tree, vector<Mat> &mat, int &idx);
		static void regularization(NetNode<Layer*>* tree, float lambda);
		static void initialize_size(NetNode<Layer*>* tree, vector<Size3> &input_size, int idx);
		static void initialize_mat(const NetNode<Layer*>* tree, vector<Size3>* mat_size);
		static void initialize_loss(NetNode<Layer*>* tree, vector<NetNode<Layer*>*> *loss);
		static void delete_layer(NetNode<Layer*>* tree);
		static void show_net(const NetNode<Layer*>* tree, std::ostream &out);
		static void forward(const NetNode<Layer*>* tree, vector<Mat> &output, int idx);
		static void forward_train(NetNode<Layer*>* tree, vector<Mat> & variable, vector<vector<Mat>> & output, int idx);
		static void back_train(NetNode<Layer*>* tree, vector<Mat>& dlayer, vector<vector<Mat>> & output, int &number, int idx);
		static void save_mat(FILE* file, const Mat &m);
		static void load_mat(FILE* file, Mat &m);

		static void nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname = "Union");
		static void generateBbox(const Mat &score, const Mat &location, std::vector<Bbox> &boundingBox_, float scale, float threshold);
		static void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
		//����
		static void resize(const Mat & src, Mat & dst, float xRatio, float yRatio, ReductionMothed mothed);
		//����
		static void resize(const Mat& src, Mat& dst, Size newSize, ReductionMothed mothed);

		static float generateGaussianNoise(float mu, float sigma);
	};
}
#endif // !__METHOD_H__
