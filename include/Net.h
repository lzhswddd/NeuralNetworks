#ifndef __NET_H__
#define __NET_H__

#include <vector>
#include "Function.h"
#include "Layer.h"
#include "Mat.h"
using std::vector;

namespace nn
{
	class Net
	{
	public:
		Net();
		~Net();
		int initialize(int input_channel);
		Size3 initialize(Size3 input_size);
		void clear();
		inline Mat forward(const Mat &input)const;

		void add(LayerInfo layerInfo);
		void addActivation(ActivationFunc act_f);
		void addMaxPool(Size poolsize, int strides);
		void addMaxPool(int pool_row, int pool_col, int strides);
		void addAveragePool(Size poolsize, int strides);
		void addAveragePool(int pool_row, int pool_col, int strides);
		void addDense(int layer_size, ActivationFunc act_f = nullptr);
		void addFullConnect(int layer_size, ActivationFunc act_f = nullptr);
		void addConv2D(int channel, int kern_size, bool is_copy_border = true, Size strides = Size(1, 1), Point anchor = Point(-1, -1), ActivationFunc act_f = nullptr);

		const Mat operator ()(const Mat &input)const;
	private:
		//权值矩阵
		vector<LayerInfo> config;
		//权值矩阵
		vector<Mat> layer;
	};
}

#endif // !__NET_H__
