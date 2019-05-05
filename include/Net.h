#ifndef __NET_H__
#define __NET_H__

#include <vector>
#include "function.h"
#include "layer.h"
#include "mat.h"
#include "trainData.h"

using std::vector;

namespace nn
{
	class Optimizer;
	class Net
	{
	public:
		Net();
		Net(string model);
		~Net();
		void clear();
		void save(string net_name)const;
		void load(string net_name);
		void save_param(string param)const;
		void load_param(string param);
		bool isEnable()const;
		vector<Size3> initialize(Size3 input_size);
		vector<Mat> forward(const Mat &input)const;
		NetNode<Layer*>* NetTree(); 
		const NetNode<Layer*>* NetTree()const;
		/**
		@param layerInfo 网络层信息
		Loss/Dense/Conv2D/Reshape/MaxPool/Dropout/Activation/AveragePool/FullConnect
		*/
		void add(Layer* layerInfo);
		/**
		@param layerInfo 网络层信息
		Loss/Dense/Conv2D/Reshape/MaxPool/Dropout/Activation/AveragePool/FullConnect
		@param layer_name 并联层名字
		layer_name="" 为在尾部并联
		*/
		void add(Layer* layerInfo, string layer_name, bool sibling = false);

		void addLoss(LossFunc loss_f, string layer_name = "");
		void addActivation(ActivationFunc act_f, string layer_name = "");
		void addMaxPool(Size poolsize, int strides, string layer_name = "");
		void addMaxPool(int pool_row, int pool_col, int strides, string layer_name = "");
		void addAveragePool(Size poolsize, int strides, string layer_name = "");
		void addAveragePool(int pool_row, int pool_col, int strides, string layer_name = "");
		void addDense(int layer_size, ActivationFunc act_f = nullptr, string layer_name = "");
		void addFullConnect(int layer_size, ActivationFunc act_f = nullptr, string layer_name = "");
		void addConv2D(int channel, int kern_size, bool is_copy_border = true, string layer_name = "", Size strides = Size(1, 1), Point anchor = Point(-1, -1), ActivationFunc act_f = nullptr);

		const vector<Mat> operator ()(const Mat &input)const;
		operator NetNode<Layer*>* ()
		{
			return &netTree;
		}
		operator const NetNode<Layer*>* ()const
		{
			return &netTree;
		}
		friend std::ostream & operator << (std::ostream &out, const Net &net);

	private:
		bool isinit = false;
		//权值矩阵
		NetNode<Layer*> netTree; 
	};
}

#endif // !__NET_H__
