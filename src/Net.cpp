#include "Net.h"
#include "method.h"
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
	netTree.child.push_back(CreateNode(Loss(loss_f, layer_name), &netTree));
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

std::ostream & nn::operator<<(std::ostream & out, const Net & net)
{
	show_net(&net.netTree, out);
	return out;
}
