#include "net.h"
#include "method.h"
#include "costfunction.h"
#include "json.hpp"
#include <fstream>
using std::ofstream;
using std::ifstream;
using json = nlohmann::json;
using namespace nn;


nn::Net::Net() : recount(new int(0))
{
	netTree = new NetNode<Layer*>(new Node(), nullptr);
}

nn::Net::Net(string model) : 
	netTree(new NetNode<Layer*>(new Node(), nullptr))
{
	load(model);
}

nn::Net::Net(const Net & net)
{
	*this = net;
}

nn::Net::~Net()
{
	if (recount != 0) {
		if (*recount == 0) {
			delete_layer(netTree);
			delete netTree;
			netTree = 0;
			delete recount;
			recount = 0;
		}
		else
			*recount -= 1;
	}
}

vector<Size3> nn::Net::initialize(Size3 input_size)
{
	vector<Size3> xsize(1);
	xsize[0] = input_size;
	netTree->child[0].data->start = true;
	initialize_size(netTree, xsize, 0);
	isinit = true;
	return xsize;
}

void nn::Net::clear()
{
	isinit = false;
	netTree->child.clear();
}

void nn::Net::save(string net_name) const
{
	ofstream file(net_name + ".json");
	FILE *data = fopen((net_name + ".param").c_str(), "wb");
	if (file.is_open()&& data) {
		json info;
		save_layer(netTree, &info, data);
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
		vector<NetNode<Layer*> *> parent(1, netTree);
		file >> info;
		for (size_t i = 0; i < info.size(); ++i)
		{
			Layer *layer = CreateLayer(info[i], data);
			if (!layer) {
				fprintf(stderr, "读取模型失败!\n");
				throw std::exception("读取模型失败!");
			}
			if ((int)parent.size() == layer->layer_index) {
				newpos->sibling.push_back(CreateNode(new Node(), newpos));
				newpos->sibling[newpos->sibling.size() - 1].child.push_back(CreateNode(layer, &newpos->sibling[newpos->sibling.size() - 1]));
				parent.push_back(&newpos->sibling[newpos->sibling.size() - 1]);
			}
			else {
				parent[layer->layer_index]->child.push_back(CreateNode(layer, parent[layer->layer_index]));
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
		save_param_layer(netTree, data);
		fclose(data);
	}
}

void nn::Net::load_param(string param)
{
	FILE *data = fopen(param.c_str(), "rb");
	if (data) {
		load_layer(netTree, data);
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
	if (netTree->child.empty())return vector<Mat>();
	vector<Mat> y(1);
	y[0] = input;
	forward(netTree, y, 0);
	return y;
}

NetNode<Layer*>* nn::Net::NetTree()
{
	return netTree;
}

const NetNode<Layer*>* nn::Net::NetTree()const
{
	return netTree;
}

void nn::Net::add(Layer* layerInfo)
{
	netTree->child.push_back(CreateNode(layerInfo, netTree));
}

void nn::Net::add(Layer* layerInfo, string layer_name, bool sibling)
{
	if (layer_name == "")
		if (sibling) {
			netTree->child[netTree->child.size() - 1].sibling.push_back(NetNode<Layer*>());
			(netTree->child[netTree->child.size() - 1].sibling.end() - 1)->child.push_back(CreateNode(layerInfo, &(netTree->child[netTree->child.size() - 1])));
		}
		else
			netTree->child[netTree->child.size() - 1].child.push_back(CreateNode(layerInfo, &(netTree->child[netTree->child.size() - 1])));
	else
		insert_layer(netTree, layer_name, layerInfo, sibling);
}

void nn::Net::addLoss(LossFunc loss_f, string layer_name)
{
	netTree->child.push_back(CreateNode(Loss(loss_f, 1, layer_name), netTree));
}

void nn::Net::addActivation(ActivationFunc act_f, string layer_name)
{
	netTree->child.push_back(CreateNode(Activation(act_f, layer_name), netTree));
}

void nn::Net::addMaxPool(Size poolsize, int strides, string layer_name)
{
	netTree->child.push_back(CreateNode(MaxPool(poolsize, strides, layer_name), netTree));
}

void nn::Net::addMaxPool(int pool_row, int pool_col, int strides, string layer_name)
{
	netTree->child.push_back(CreateNode(MaxPool(Size(pool_row, pool_col), strides, layer_name), netTree));
}

void nn::Net::addAveragePool(Size poolsize, int strides, string layer_name)
{
	netTree->child.push_back(CreateNode(AveragePool(poolsize, strides, layer_name), netTree));
}

void nn::Net::addAveragePool(int pool_row, int pool_col, int strides, string layer_name)
{
	netTree->child.push_back(CreateNode(AveragePool(Size(pool_row, pool_col), strides, layer_name), netTree));
}

void nn::Net::addDense(int layer_size, ActivationFunc act_f, string layer_name)
{
	netTree->child.push_back(CreateNode(Dense(layer_size, act_f, layer_name), netTree));
}

void nn::Net::addFullConnect(int layer_size, ActivationFunc act_f, string layer_name)
{
	netTree->child.push_back(CreateNode(FullConnect(layer_size, act_f, layer_name), netTree));
}

void nn::Net::addConv2D(int channel, int kern_size, bool is_copy_border, string layer_name, Size strides, Point anchor, ActivationFunc act_f)
{
	netTree->child.push_back(CreateNode(Conv2D(channel, kern_size, is_copy_border, act_f, layer_name, strides, anchor), netTree));
}

const vector<Mat> nn::Net::operator()(const Mat & input) const
{
	return forward(input);
}

Net& nn::Net::operator=(const Net &net)
{
	if (this == &net)
		return *this;
	if (recount != 0) {
		if (*recount == 0) {
			delete_layer(netTree);
			delete netTree;
			netTree = 0;
			delete recount;
			recount = 0;
		}
		else
			*recount -= 1;
	}
	recount = net.recount;
	if (recount != 0)
		*recount += 1;
	isinit = net.isinit;
	netTree = net.netTree;
	return *this;
}

std::ostream & nn::operator <<(std::ostream & out, const Net & net)
{
	Net::show_net(net.netTree, out);
	return out;
}

void nn::Net::show_net(const NetNode<Layer*>* tree, std::ostream & out)
{
	for (const NetNode<Layer*> & layerInfo : tree->child) {
		const Layer *layer = layerInfo.data;
		out << layer << std::endl;
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				show_net(&layerInfo.sibling[i], out);
			}
		}
		if (!layerInfo.child.empty()) {
			show_net(&layerInfo, out);
		}
	}
}

NetNode<Layer*> nn::Net::CreateNode(Layer* data, NetNode<Layer*>* parent)
{
	NetNode<Layer*> netnode(data, parent);
	if (parent)
		netnode.data->layer_index = parent->data->layer_index;
	return netnode;
}
void nn::Net::save_layer(const NetNode<Layer*>* tree, json *j, FILE *data)
{
	for (const NetNode<Layer*> & node : tree->child)
	{
		node.data->save(j, data);
		if (!node.sibling.empty()) {
			for (const NetNode<Layer*> & brother : node.sibling) {
				save_layer(&brother, j, data);
			}
		}
		if (!node.child.empty()) {
			save_layer(&node, j, data);
		}
	}
}
void nn::Net::save_param_layer(const NetNode<Layer*>* tree, FILE * data)
{
	for (const NetNode<Layer*> & node : tree->child) {
		if (node.data->isparam)
			((parametrics*)node.data)->save_param(data);
		if (!node.sibling.empty()) {
			for (const NetNode<Layer*> & brother : node.sibling) {
				save_param_layer(&brother, data);
			}
		}
		if (!node.child.empty()) {
			save_param_layer(&node, data);
		}
	}
}
void nn::Net::load_layer(NetNode<Layer*>* tree, FILE * data)
{
	for (NetNode<Layer*> & node : tree->child) {
		if (node.data->isparam)
			((parametrics*)node.data)->load_param(data);
		if (!node.sibling.empty()) {
			for (NetNode<Layer*> & brother : node.sibling) {
				load_layer(&brother, data);
			}
		}
		if (!node.child.empty()) {
			load_layer(&node, data);
		}
	}
}
int nn::Net::insert_layer(NetNode<Layer*>* tree, string name, Layer *layer, bool sibling)
{
	int success_insert = 0;
	for (vector<NetNode<Layer*>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		if (!node->sibling.empty()) {
			for (NetNode<Layer*> & brother : node->sibling) {
				success_insert += insert_layer(&brother, name, layer, sibling);
			}
		}
		Layer *layerInfo = node->data;
		if (layerInfo->name == name) {
			if (sibling) {
				Layer *total = (Layer*)new Node();
				node->sibling.push_back(NetNode<Layer*>(total, nullptr));
				node->sibling[node->sibling.size() - 1].data->layer_index = node->data->layer_index + 1;
				NetNode<Layer*> netnode = CreateNode(layer, &node->sibling[node->sibling.size() - 1]);
				node->sibling[node->sibling.size() - 1].child.push_back(netnode);
			}
			else {
				node = node->parent->child.insert(node + 1, CreateNode(layer, node->parent));
			}
			success_insert += 1;
		}
		if (!node->child.empty()) {
			success_insert += insert_layer(&*node, name, layer, sibling);
		}
	}
	return success_insert;
}
void nn::Net::update_layer(NetNode<Layer*>* tree, vector<Mat>& mat, int &idx)
{
	for (NetNode<Layer*> & node : tree->child) {
		if (node.data->isupdate) {
			((parametrics*)node.data)->update(mat, &idx);
		}
		if (!node.sibling.empty()) {
			for (NetNode<Layer*> & brother : node.sibling) {
				update_layer(&brother, mat, idx);
			}
		}
		if (!node.child.empty()) {
			update_layer(&node, mat, idx);
		}
	}
}
void nn::Net::regularization(NetNode<Layer*>* tree, float lambda)
{
	for (NetNode<Layer*> & node : tree->child) {
		if (node.data->isparam) {
			parametrics* p = ((parametrics*)node.data);
			p->regularization = true;
			p->lambda = lambda;
			p->updateregular();
		}
		if (!node.sibling.empty()) {
			for (NetNode<Layer*> & brother : node.sibling) {
				regularization(&brother, lambda);
			}
		}
		if (!node.child.empty()) {
			regularization(&node, lambda);
		}
	}
}
void nn::Net::initialize_loss(NetNode<Layer*>* tree, vector<NetNode<Layer*>*>* loss)
{
	for (vector<NetNode<Layer*>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		Layer *layerInfo = node->data;
		switch (layerInfo->type)
		{
		case nn::LOSS:
			if (loss) {
				loss->push_back(&*node);
				(node - 1)->data->last = ((costfunction*)node->data)->ignore_active;
				/*if (tree->child.size() > 1) {
					node--;
					tree->child.pop_back();
				}
				else {
					tree->child.pop_back();
					return;
				}*/
			}
			break;
		default:
			break;
		}
		if (!node->sibling.empty()) {
			for (NetNode<Layer*> & brother : node->sibling) {
				initialize_loss(&brother, loss);
			}
		}
		if (!node->child.empty()) {
			initialize_loss(&*node, loss);
		}
	}
}
void nn::Net::initialize_size(NetNode<Layer*>* tree, vector<Size3>& input_size, int idx)
{
	for (vector<NetNode<Layer*>>::iterator node = tree->child.begin(); node != tree->child.end();) {
		if (node->data->type == FULLCONNECTION)
		{
			if (input_size[idx].c != 1 || input_size[idx].w != 1)
			{
				if (node->data->start) {
					node->data->start = false;
					node = tree->child.insert(node, CreateNode(Reshape(Size3(input_size[idx].h*input_size[idx].w*input_size[idx].c, 1, 1)), &(*node)));
					node->data->start = true;
				}
				else {
					node = tree->child.insert(node, CreateNode(Reshape(Size3(input_size[idx].h*input_size[idx].w*input_size[idx].c, 1, 1)), &(*node)));
				}
				continue;
			}
		}
		input_size[idx] = node->data->initialize(input_size[idx]);
		if (!node->sibling.empty()) {
			if (!node->sibling.empty()) {
				for (int i = 0; i < node->sibling.size(); ++i) {
					input_size.push_back(input_size[idx]);
					initialize_size(&node->sibling[i], input_size, idx + 1 + i);
				}
			}
		}
		if (!node->child.empty()) {
			initialize_size(&*node, input_size, idx);
		}
		++node;
	}
}
void nn::Net::initialize_mat(const NetNode<Layer*>* tree, vector<Size3>* mat_size)
{
	for (const NetNode<Layer*> & node : tree->child) {
		if (node.data->isparam)
			((parametrics*)node.data)->append_size(mat_size);
		if (!node.sibling.empty()) {
			for (const NetNode<Layer*> brother : node.sibling)
				initialize_mat(&brother, mat_size);
		}
		if (!node.child.empty()) {
			initialize_mat(&node, mat_size);
		}
	}
}
void nn::Net::delete_layer(NetNode<Layer*>* tree)
{
	for (NetNode<Layer*> &node : tree->child) {
		if (!node.sibling.empty()) {
			for (NetNode<Layer*> & brother : node.sibling) {
				delete_layer(&brother);
			}
			delete node.data;
			node.data = nullptr;
		}
		if (!node.child.empty()) {
			delete_layer(&node);
		}
	}
}
void nn::Net::forward(const NetNode<Layer*>* tree, vector<Mat>& output, int idx)
{
	for (const NetNode<Layer*> & layerInfo : tree->child) {
		const Layer *layer = layerInfo.data;
		if (layer->isforword) {
			layer->forword(output[idx], output[idx]);
		}
		if (!layerInfo.sibling.empty()) {
			for

				(int i = 0; i < layerInfo.sibling.size(); ++i) {
				output.push_back(output[idx]);
				forward(&layerInfo.sibling[i], output, idx + 1 + i);
			}
		}
		if (!layerInfo.child.empty()) {
			forward(&layerInfo, output, idx);
		}
	}
}
void nn::Net::forward_train(NetNode<Layer*>* tree, vector<Mat> & variable, vector<vector<Mat>>& output, int idx)
{
	for (NetNode<Layer*> & layerInfo : tree->child) {
		Layer *layer = layerInfo.data;
		layer->forword_train(output[idx], output[idx], variable);
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				output.push_back(output[idx]);
				vector<Mat> v = variable;
				forward_train(&layerInfo.sibling[i], v, output, idx + 1 + i);
			}
		}
		if (!layerInfo.child.empty()) {
			forward_train(&layerInfo, variable, output, idx);
		}
	}
}
void nn::Net::back_train(NetNode<Layer*>* tree, vector<Mat>& dlayer, vector<vector<Mat>> & output, int &number, int idx)
{
	for (int index = (int)tree->child.size() - 1; index >= 0; --index) {
		if (!tree->child[index].sibling.empty()) {
			for (int i = 0; i < tree->child[index].sibling.size(); ++i) {
				back_train(&tree->child[index].sibling[i], dlayer, output, number, idx + i + 1);
				for (size_t j = 0; j < output[idx].size(); ++j)
					output[idx][j] += output[idx + i + 1][j];
			}
		}
		Layer *layer = tree->child[index].data;
		if (layer->isback && !layer->start) {
			layer->back(output[idx], output[idx], &dlayer, &number);
		}
		if (!tree->child[index].child.empty()) {
			back_train(&tree->child[index], dlayer, output, number, idx);
		}
	}
}