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
		@brief CreateMat �����������
		��СΪrow*col*channel, Ԫ�ط�Χ[low, top]
		@param row ��������
		@param col ��������
		@param channel ����ͨ����
		@param low ����
		@param top ����
		*/
	const Mat CreateMat(int row, int col, int channel = 1, double low = -0.5, double top = 0.5);
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
	const Mat CreateMat(int row, int col, int channel_input, int channel_output, double low = -0.5, double top = 0.5);
	const Mat conv2d(const Mat& input, const Mat& kern, const Size& strides, Point anchor, bool is_copy_border = true);
	const Mat MaxPool(const Mat & input, const Size & ksize, int stride);
	const Mat AveragePool(const Mat& input, const Size & ksize, int stride);
	const Mat FullConnection(const Mat & input, const Mat & layer, const Mat & bias);
}
#endif // !__METHOD_H__
