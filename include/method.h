#ifndef __METHOD_H__
#define __METHOD_H__

#include "Mat.h"

using nn::Mat;
using nn::Size3;
using nn::Size;
using nn::Point;

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
	const Mat CreateMat(int row, int col, int channel = 1, double low = -0.5, double top = 0.5);
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
	const Mat CreateMat(int row, int col, int channel_input, int channel_output, double low = -0.5, double top = 0.5);
	const Mat conv2d(const Mat& input, const Mat& kern, const Size& strides, Point anchor, bool is_copy_border = true);
	const Mat MaxPool(const Mat & input, const Size & ksize, int stride);
	const Mat AveragePool(const Mat& input, const Size & ksize, int stride);
	const Mat FullConnection(const Mat & input, const Mat & layer, const Mat & bias);
}
#endif // !__METHOD_H__
