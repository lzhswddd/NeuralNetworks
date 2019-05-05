#include "net.h"
#include "method.h"
#include "costfunction.h"
#include "json.hpp"
#include <fstream>
using std::ofstream;
using std::ifstream;
using json = nlohmann::json;
using namespace nn;


nn::Net::Net()
{
	netTree.data = new Node();
	netTree.parent = nullptr;
}

nn::Net::Net(string model)
{
	load(model);
}

nn::Net::~Net()
{
	method::delete_layer(&netTree);
}

vector<Size3> nn::Net::initialize(Size3 input_size)
{
	vector<Size3> xsize(1);
	xsize[0] = input_size;
	netTree.child[0].data->start = true;
	method::initialize_size(&netTree, xsize, 0);
	isinit = true;
	return xsize;
}

void nn::Net::clear()
{
	isinit = false;
	netTree.child.clear();
}

void nn::Net::save(string net_name) const
{
	ofstream file(net_name + ".json");
	FILE *data = fopen((net_name + ".param").c_str(), "wb");
	if (file.is_open()&& data) {
		json info;
		method::save_layer(&netTree, &info, data);
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
		NetNode<Layer*> *newpos;
		vector<NetNode<Layer*> *> parent(1, &netTree);
		file >> info;
		for (size_t i = 0; i < info.size(); ++i)
		{
			Layer *layer = CreateLayer(info[i], data);
			if (!layer) {
				fprintf(stderr, "读取模型失败!\n");
				throw std::exception("读取模型失败!");
			}
			if ((int)parent.size() == layer->layer_index) {
				newpos->sibling.push_back(method::CreateNode(new Node(), newpos));
				newpos->sibling[newpos->sibling.size() - 1].child.push_back(method::CreateNode(layer, &newpos->sibling[newpos->sibling.size() - 1]));
				parent.push_back(&newpos->sibling[newpos->sibling.size() - 1]);
			}
			else {
				parent[layer->layer_index]->child.push_back(method::CreateNode(layer, parent[layer->layer_index]));
			}
			newpos = &parent[layer->layer_index]->child[parent[layer->layer_index]->child.size() - 1];
			std::cout << info[i].dump(4) << std::endl;
		}
		file.close(); fclose(data);
	}
	isinit = true;
}

void nn::Net::save_param(string param) const
{
	FILE *data = fopen(param.c_str(), "wb");
	if (data) {
		method::save_param_layer(&netTree, data);
		fclose(data);
	}
}

void nn::Net::load_param(string param)
{
	FILE *data = fopen(param.c_str(), "rb");
	if (data) {
		method::load_layer(&netTree, data);
		fclose(data);
	}
}

bool nn::Net::isEnable() const
{
	return isinit;
}

vector<Mat> nn::Net::forward(const Mat & input) const
{
	if (!isinit) {
		fprintf(stderr, "error: 网络未初始化!\n");
		return vector<Mat>();
	}
	if (netTree.child.empty())return vector<Mat>();
	vector<Mat> y(1);
	y[0] = input;
	method::forward(&netTree, y, 0);
	return y;
}

NetNode<Layer*>* nn::Net::NetTree()
{
	return &netTree;
}

const NetNode<Layer*>* nn::Net::NetTree()const
{
	return &netTree;
}

void nn::Net::add(Layer* layerInfo)
{
	netTree.child.push_back(method::CreateNode(layerInfo, &netTree));
}

void nn::Net::add(Layer* layerInfo, string layer_name, bool sibling)
{
	if (layer_name == "")
		if (sibling) {
			netTree.child[netTree.child.size() - 1].sibling.push_back(NetNode<Layer*>());
			(netTree.child[netTree.child.size() - 1].sibling.end() - 1)->child.push_back(method::CreateNode(layerInfo, &(netTree.child[netTree.child.size() - 1])));
		}
		else
			netTree.child[netTree.child.size() - 1].child.push_back(method::CreateNode(layerInfo, &(netTree.child[netTree.child.size() - 1])));
	else
		method::insert_layer(&netTree, layer_name, layerInfo, sibling);
}

void nn::Net::addLoss(LossFunc loss_f, string layer_name)
{
	netTree.child.push_back(method::CreateNode(Loss(loss_f, 1, layer_name), &netTree));
}

void nn::Net::addActivation(ActivationFunc act_f, string layer_name)
{
	netTree.child.push_back(method::CreateNode(Activation(act_f, layer_name), &netTree));
}

void nn::Net::addMaxPool(Size poolsize, int strides, string layer_name)
{
	netTree.child.push_back(method::CreateNode(MaxPool(poolsize, strides, layer_name), &netTree));
}

void nn::Net::addMaxPool(int pool_row, int pool_col, int strides, string layer_name)
{
	netTree.child.push_back(method::CreateNode(MaxPool(Size(pool_row, pool_col), strides, layer_name), &netTree));
}

void nn::Net::addAveragePool(Size poolsize, int strides, string layer_name)
{
	netTree.child.push_back(method::CreateNode(AveragePool(poolsize, strides, layer_name), &netTree));
}

void nn::Net::addAveragePool(int pool_row, int pool_col, int strides, string layer_name)
{
	netTree.child.push_back(method::CreateNode(AveragePool(Size(pool_row, pool_col), strides, layer_name), &netTree));
}

void nn::Net::addDense(int layer_size, ActivationFunc act_f, string layer_name)
{
	netTree.child.push_back(method::CreateNode(Dense(layer_size, act_f, layer_name), &netTree));
}

void nn::Net::addFullConnect(int layer_size, ActivationFunc act_f, string layer_name)
{
	netTree.child.push_back(method::CreateNode(FullConnect(layer_size, act_f, layer_name), &netTree));
}

void nn::Net::addConv2D(int channel, int kern_size, bool is_copy_border, string layer_name, Size strides, Point anchor, ActivationFunc act_f)
{
	netTree.child.push_back(method::CreateNode(Conv2D(channel, kern_size, is_copy_border, act_f, layer_name, strides, anchor), &netTree));
}

const vector<Mat> nn::Net::operator()(const Mat & input) const
{
	return forward(input);
}

std::ostream & nn::operator <<(std::ostream & out, const Net & net)
{
	method::show_net(&net.netTree, out);
	return out;
}
