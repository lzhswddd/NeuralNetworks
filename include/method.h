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
		static const Mat Xavier(int row, int col, int channel, int n1, int n2);
		/**
		@brief CreateMat 创建随机矩阵
		大小为row*col*channel, 元素范围[low, top]
		@param row 矩阵行数
		@param col 矩阵列数
		@param channel 矩阵通道数
		@param low 下限
		@param top 上限
		*/
		static const Mat Random(int row, int col, int channel = 1);
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
		static const Mat Random(int row, int col, int channel_input, int channel_output);
		
		static void save_mat(FILE* file, const Mat &m);
		static void load_mat(FILE* file, Mat &m);

		static void nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname = "Union");
		static void generateBbox(const Mat &score, const Mat &location, std::vector<Bbox> &boundingBox_, float scale, float threshold);
		static void refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square);
		//缩放
		static void resize(const Mat & src, Mat & dst, float xRatio, float yRatio, ReductionMothed mothed);
		//缩放
		static void resize(const Mat& src, Mat& dst, Size newSize, ReductionMothed mothed);

		static float generateGaussianNoise(float mu, float sigma);
	};
}
#endif // !__METHOD_H__
