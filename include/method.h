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
