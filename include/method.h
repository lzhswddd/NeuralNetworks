#ifndef __METHOD_H__
#define __METHOD_H__

#include "Layer.h"

using nn::Mat;
using nn::Size3;
using nn::Size;
using nn::Point;
using nn::ConvInfo;

namespace nn
{
	/**
	@brief CreateMat 创建随机矩阵
	大小为row*col*channel, 元素范围[low, top]
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel 矩阵通道数
	@param low 下限
	@param top 上限
	*/
	const Mat CreateMat(int row, int col, int channel = 1, float low = -0.5, float top = 0.5);
	/**
	@brief CreateMat 创建随机矩阵
	大小为row*col*(channel_output - channel_input), 元素范围[low, top]
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel_input 输入矩阵通道数
	@param channel_output 输出矩阵通道数
	@param low 下限
	@param top 上限
	*/
	const Mat CreateMat(int row, int col, int channel_input, int channel_output, float low = -0.5, float top = 0.5);
	const Mat iconv2d(const Mat& input, const Mat& kern, const Size3 & E_kern_size, ConvInfo conv, bool is_copy_border = true);
	const Mat conv2d(const Mat& input, const Mat& kern, const Size& strides, Point anchor, bool is_copy_border = true);
	const Mat upsample(const Mat & input, Size ksize, int stride, const Mat & markpoint = Mat());
	const Mat MaxPool(const Mat & input, Size ksize, int stride);
	const Mat MaxPool(const Mat & input, Mat & markpoint, Size ksize, int stride);
	const Mat iMaxPool(const Mat & input, const Mat & markpoint, Size ksize, int stride);
	const Mat iAveragePool(const Mat& input, Size ksize, int stride);
	const Mat AveragePool(const Mat& input, Size ksize, int stride);
	const Mat FullConnection(const Mat & input, const Mat & layer, const Mat & bias);

	NetNode<Layer> CreateNode(const Layer &data, NetNode<Layer>* parent);
	int insert_layer(NetNode<Layer>* tree, string name, Layer &layer, bool sibling);
	void update_layer(NetNode<Layer>* tree, vector<Mat> &mat, int idx);
	void initialize_loss(NetNode<Layer>* tree, vector<NetNode<Layer>*> *loss);
	void initialize_size(NetNode<Layer>* tree, vector<Size3> &input_size, int idx);
	void initialize_channel(NetNode<Layer>* tree, vector<int> *input_channel, int idx);
	void initialize_mat(const NetNode<Layer>* tree, vector<Size3>* mat_size);
	void show_net(const NetNode<Layer>* tree, std::ostream &out);
	void forward(const NetNode<Layer>* tree, vector<Mat> &output, int idx);
	void forward_train(const NetNode<Layer>* tree, vector<vector<Mat>> &variable, vector<Mat> &output, int idx);
	void back_train(const NetNode<Layer>* tree, vector<Mat> &dlayer, vector<vector<Mat>> &x, vector<Mat> &output, int idx);
}
#endif // !__METHOD_H__
