#ifndef __LAYER_H__
#define __LAYER_H__

#include "NetParam.h"
#include <fstream>

namespace nn
{
	class Layer
	{
	public:
		explicit Layer(string name = "") : name(name), type(NONE), param(), bias(), layer_index(0){}
		string name;
		LayerType type;
		PoolInfo pInfo;
		FCInfo fcInfo; 
		ConvInfo convInfo;
		DropoutInfo dropoutInfo;
		ReShapeInfo reshapeInfo;
		ActivationInfo active;
		LossInfo loss;
		int layer_index;
		bool last = false;
		Mat param;
		Mat bias;
		static string Type2String(LayerType type);
		static LayerType String2Type(string str);
		friend std::ostream & operator << (std::ostream &out, const Layer &layer);
	};

	Layer Reshape(Size3 size, string name = "");
	Layer Loss(LossFunc loss_f, float weight = 1, string name = "");
	Layer Loss(ReduceType loss_f, float weight = 1, string name = "");
	Layer Loss(LossFunc f, LossFunc df, bool ignore_activate, float weight = 1, string name = "");
	Layer Dropout(float dropout, string name = "");
	Layer Activation(ActivationFunc acti, string name = "");
	Layer Activation(ActivationType acti, string name = "");
	Layer Activation(ActivationFunc f, ActivationFunc df, string name = "");
	Layer MaxPool(Size poolsize, int strides, string name = "");
	Layer AveragePool(Size poolsize, int strides, string name = "");
	Layer Dense(int layer_size, ActivationFunc act_f = nullptr, string name = "");
	Layer FullConnect(int layer_size, ActivationFunc act_f = nullptr, string name = "");
	Layer Conv2D(ConvInfo cinfo, ActivationFunc act_f = nullptr, string name = "");
	Layer Conv2D(int output_channel, int kern_size, bool is_copy_border = true, ActivationFunc act_f = nullptr, string name = "", Size strides = Size(1, 1), Point anchor = Point(-1, -1));
}

#endif // !__LAYER_H__
