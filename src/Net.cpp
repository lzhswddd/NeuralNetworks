#include "Net.h"
#include "method.h"
#include "json.hpp"
#include <fstream>
using std::ofstream;
using std::ifstream;
using json = nlohmann::json;
using namespace nn;


Net::Net()
{
	netTree.data = Layer();
	netTree.parent = nullptr;
}

Net::~Net()
{
}

vector<int> nn::Net::initialize(int input_channel)
{
	vector<int> channel(1);
	channel[0] = input_channel;
	initialize_channel(&netTree, &channel, 0);
	return channel;
}

vector<Size3> nn::Net::initialize(Size3 input_size)
{
	vector<Size3> xsize(1);
	xsize[0] = input_size;
	initialize_size(&netTree, xsize, 0);
	return xsize;
}

void nn::Net::clear()
{
	netTree.child.clear();
}

void nn::Net::save(string net_name) const
{
	ofstream file(net_name + ".json");
	FILE *data = fopen((net_name + ".param").c_str(), "wb");
	if (file.is_open()&& data) {
		json info;
		save_layer(&netTree, &info, data);
		file << info.dump(4) << std::endl;
		file.close(); fclose(data);
	}
}

void nn::Net::load(string net_name)
{
	ifstream file(net_name + ".json");
	FILE *data = fopen((net_name + ".param").c_str(), "rb");
	if (file.is_open() && data) {
		json info;
		NetNode<Layer> *newpos;
		vector<NetNode<Layer> *> parent(1, &netTree);
		file >> info;
		for (size_t i = 0; i < info.size(); ++i)
		{
			Layer layer(info[i]["name"]);
			layer.type = Layer::String2Type(info[i]["type"]);
			layer.layer_index = info[i]["layer"];
			layer.last = info[i]["last"];
			switch (layer.type)
			{
			case nn::NONE:
				break;
			case nn::CONV2D:
				layer.convInfo.anchor.x = info[i]["anchor"]["x"];
				layer.convInfo.anchor.y = info[i]["anchor"]["y"];
				layer.convInfo.channel = info[i]["channel"];
				layer.convInfo.kern_size = info[i]["kernSize"];
				layer.convInfo.is_copy_border = info[i]["is_copy_border"];
				layer.convInfo.strides.hei = info[i]["strides"]["height"];
				layer.convInfo.strides.wid = info[i]["strides"]["width"];
				layer.convInfo.isact = info[i]["isact"];
				if (layer.convInfo.isact)
					SetFunc(info[i]["activate"], &layer.active.f, &layer.active.df);
				break;
			case nn::MAX_POOL:
				layer.pInfo.strides = info[i]["strides"];
				layer.pInfo.size.hei = info[i]["size"]["height"];
				layer.pInfo.size.wid = info[i]["size"]["width"];
				break;
			case nn::AVERAGE_POOL:
				layer.pInfo.strides = info[i]["strides"];
				layer.pInfo.size.hei = info[i]["size"]["height"];
				layer.pInfo.size.wid = info[i]["size"]["width"];
				break;
			case nn::FULLCONNECTION:
				layer.fcInfo.isact = info[i]["isact"];
				if (layer.fcInfo.isact)
					SetFunc(info[i]["activate"], &layer.active.f, &layer.active.df);
				layer.fcInfo.size = info["size"];
				break;
			case nn::ACTIVATION:
				SetFunc(info[i]["activate"], &layer.active.f, &layer.active.df);
				break;
			case nn::RESHAPE:
				layer.reshapeInfo.size.x = info[i]["size"]["x"];
				layer.reshapeInfo.size.y = info[i]["size"]["y"];
				layer.reshapeInfo.size.z = info[i]["size"]["z"];
				break;
			case nn::DROPOUT:
				layer.dropoutInfo.dropout = info[i]["dropout"];
				break;
			case nn::LOSS:
				SetFunc(info[i]["loss"], &layer.loss.f, &layer.loss.df);
				layer.loss.ignore_active = info[i]["ignore_active"];
				break;
			default:
				continue;
			}
			if (info[i]["matrix"]) {
				int param[3]; int bias[3];
				fread(param, sizeof(int) * 3, 1, data);
				layer.param = zeros(param[0], param[1], param[2]);
				fread(layer.param, sizeof(float)*layer.param.length(), 1, data);
				fread(bias, sizeof(int) * 3, 1, data);
				layer.bias = zeros(bias[0], bias[1], bias[2]);
				fread(layer.bias, sizeof(float)*layer.bias.length(), 1, data);
			}
			if ((int)parent.size() == layer.layer_index) {
				newpos->sibling.push_back(CreateNode(Layer(), newpos));
				newpos->sibling[newpos->sibling.size() - 1].child.push_back(CreateNode(layer, &newpos->sibling[newpos->sibling.size() - 1]));
				parent.push_back(&newpos->sibling[newpos->sibling.size() - 1]);
			}
			else {
				parent[layer.layer_index]->child.push_back(CreateNode(layer, parent[layer.layer_index]));
			}
			newpos = &parent[layer.layer_index]->child[parent[layer.layer_index]->child.size() - 1];
			std::cout << info[i].dump(4) << std::endl;
		}
		file.close(); fclose(data);
	}
}

void nn::Net::save_param(string param) const
{
	FILE *data = fopen(param.c_str(), "wb");
	if (data) {
		save_param_layer(&netTree, data);
		fclose(data);
	}
}

void nn::Net::load_param(string param)
{
	FILE *data = fopen(param.c_str(), "rb");
	if (data) {
		load_layer(&netTree, data);
		fclose(data);
	}
}

vector<Mat> nn::Net::forward(const Mat & input) const
{
	if (netTree.child.empty())return vector<Mat>();
	vector<Mat> y(1);
	y[0] = input;
	nn::forward(&netTree, y, 0);
	return y;
}

NetNode<Layer>* nn::Net::NetTree()
{
	return &netTree;
}

const NetNode<Layer>* nn::Net::NetTree()const
{
	return &netTree;
}

void nn::Net::add(Layer layerInfo)
{
	netTree.child.push_back(CreateNode(layerInfo, &netTree));
}

void nn::Net::add(Layer layerInfo, string layer_name, bool sibling)
{
	if (layer_name == "")
		if (sibling) {
			netTree.child[netTree.child.size() - 1].sibling.push_back(NetNode<Layer>());
			(netTree.child[netTree.child.size() - 1].sibling.end() - 1)->child.push_back(CreateNode(layerInfo, &(netTree.child[netTree.child.size() - 1])));
		}
		else
			netTree.child[netTree.child.size() - 1].child.push_back(CreateNode(layerInfo, &(netTree.child[netTree.child.size() - 1])));
	else
		insert_layer(&netTree, layer_name, layerInfo, sibling);
}

void nn::Net::addLoss(LossFunc loss_f, string layer_name)
{
	netTree.child.push_back(CreateNode(Loss(loss_f, 1, layer_name), &netTree));
}

void nn::Net::addActivation(ActivationFunc act_f, string layer_name)
{
	netTree.child.push_back(CreateNode(Activation(act_f, layer_name), &netTree));
}

void nn::Net::addMaxPool(Size poolsize, int strides, string layer_name)
{
	netTree.child.push_back(CreateNode(MaxPool(poolsize, strides, layer_name), &netTree));
}

void nn::Net::addMaxPool(int pool_row, int pool_col, int strides, string layer_name)
{
	netTree.child.push_back(CreateNode(MaxPool(Size(pool_row, pool_col), strides, layer_name), &netTree));
}

void nn::Net::addAveragePool(Size poolsize, int strides, string layer_name)
{
	netTree.child.push_back(CreateNode(AveragePool(poolsize, strides, layer_name), &netTree));
}

void nn::Net::addAveragePool(int pool_row, int pool_col, int strides, string layer_name)
{
	netTree.child.push_back(CreateNode(AveragePool(Size(pool_row, pool_col), strides, layer_name), &netTree));
}

void nn::Net::addDense(int layer_size, ActivationFunc act_f, string layer_name)
{
	netTree.child.push_back(CreateNode(Dense(layer_size, act_f, layer_name), &netTree));
}

void nn::Net::addFullConnect(int layer_size, ActivationFunc act_f, string layer_name)
{
	netTree.child.push_back(CreateNode(FullConnect(layer_size, act_f, layer_name), &netTree));
}

void nn::Net::addConv2D(int channel, int kern_size, bool is_copy_border, string layer_name, Size strides, Point anchor, ActivationFunc act_f)
{
	netTree.child.push_back(CreateNode(Conv2D(channel, kern_size, is_copy_border, act_f, layer_name, strides, anchor), &netTree));
}

const vector<Mat> nn::Net::operator()(const Mat & input) const
{
	return forward(input);
}

std::ostream & nn::operator <<(std::ostream & out, const Net & net)
{
	show_net(&net.netTree, out);
	return out;
}
