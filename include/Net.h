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
		Net(const Net &net);
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
			return netTree;
		}
		operator const NetNode<Layer*>* ()const
		{
			return netTree;
		}
		Net& operator = (const Net &net);
		friend std::ostream & operator << (std::ostream &out, const Net &net);

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
	private:
		bool isinit = false;
		int *recount = 0;
		//权值矩阵
		NetNode<Layer*> *netTree; 
	};
}

#endif // !__NET_H__
