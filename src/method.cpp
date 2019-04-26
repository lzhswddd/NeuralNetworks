#include "method.h"
#include "json.hpp"
#include <iostream>
#include <fstream>
using json = nlohmann::json;

using namespace nn;

const Mat nn::CreateMat(int row, int col, int channel, float low, float top)
{
	return mRand(0, int(top - low), row, col, channel, true) + low;
}
const Mat nn::CreateMat(int row, int col, int channel_input, int channel_output, float low, float top)
{
	return mRand(0, int(top - low), row, col, channel_output * channel_input, true) + low;
}
const Mat nn::iconv2d(const Mat & input, const Mat & kern, const Size3 & E_kern_size, ConvInfo conv, bool is_copy_border)
{
	Mat output;
	Size3 area;
	int left, right, top, bottom;
	if (E_kern_size.x == 0 || E_kern_size.y == 0 || E_kern_size.z == 0) {
		area.z = kern.channels() / input.channels();
		if (conv.is_copy_border) {
			area.x = input.rows() * conv.strides.hei;
			area.y = input.cols() * conv.strides.wid;
			output = zeros(input.rows() * conv.strides.hei, input.cols() * conv.strides.wid, area.z);
		}
		else {
			if (conv.anchor == Point(-1, -1)) {
				conv.anchor.x = kern.rows() % 2 ? kern.rows() / 2 : kern.rows() / 2 - 1;
				conv.anchor.y = kern.cols() % 2 ? kern.cols() / 2 : kern.cols() / 2 - 1;
			}
			top = conv.anchor.x;
			bottom = kern.rows() - conv.anchor.x - 1;
			left = conv.anchor.y;
			right = kern.cols() - conv.anchor.y - 1;
			area.x = (input.rows() + top + bottom) * conv.strides.hei;
			area.y = (input.cols() + left + right) * conv.strides.wid;
			output = zeros(area);
		}
		for (int i = 0; i < area.z; i++) {
			Mat sum = zeros(area.x, area.y);
			for (int j = 0; j < input.channels(); j++) {
				if (!conv.is_copy_border) {
					Mat copy_border = copyMakeBorder(input[j], top, bottom, left, right);
					sum += Filter2D(copy_border, kern[i*input.channels() + j], conv.anchor, conv.strides, is_copy_border);
				}
				else
					sum += Filter2D(input[j], kern[i*input.channels() + j], conv.anchor, conv.strides, is_copy_border);
			}
			//output.mChannel(sum / float(input.channels()), i);
			output.mChannel(sum, i);
		}

	}
	else {
		area.z = E_kern_size.z;
		output = zeros(E_kern_size.x, E_kern_size.y, area.z);
		for (int i = 0; i < kern.channels(); i++) {
			for (int j = 0; j < input.channels(); j++) {
				if (conv.is_copy_border) {
					Mat copy_border = copyMakeBorder(input[j], E_kern_size.x / 2, E_kern_size.x / 2, E_kern_size.y / 2, E_kern_size.y / 2);
					output.mChannel(Filter2D(copy_border, kern[i], conv.anchor, conv.strides, is_copy_border), i*input.channels() + j);
				}
				else
					output.mChannel(Filter2D(input[j], kern[i], conv.anchor, conv.strides, is_copy_border), i*input.channels() + j);
			}
		}
	}
	return output;
}const Mat nn::conv2d(const Mat & input, const Mat & kern, const Size & strides, Point anchor, bool is_copy_border)
{
	Mat output;
	if (is_copy_border) {
		output = zeros(input.rows(), input.cols(), kern.channels() / input.channels());
	}
	else {
		int left, right, top, bottom;
		Size3 size = mCalSize(input.size3(), kern.size3(), anchor, strides, left, right, top, bottom);
		output = zeros(size);
	}
	for (int i = 0; i < kern.channels() / input.channels(); i++) {
		Mat sum = zeros(output.rows(), output.cols());
		for (int j = 0; j < input.channels(); j++) {
			sum += Filter2D(input[j], kern[i*input.channels() + j], anchor, strides, is_copy_border);
		}
		output.mChannel(sum, i);
	}
	return output;
}
const Mat nn::upsample(const Mat & input, Size ksize, int stride, const Mat & markpoint)
{
	if (markpoint.empty())
		return iAveragePool(input, ksize, stride);
	else
		return iMaxPool(input, markpoint, ksize, stride);
}
const Mat nn::MaxPool(const Mat & input, Size ksize, int stride)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				float value = input(0, 0, z);
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
						}
					}
				}
				dst(row, col, z) = value;
			}
	return dst;
}
const Mat nn::MaxPool(const Mat & input, Mat & markpoint, Size ksize, int stride)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	markpoint.create(dst.rows()*dst.cols(), 2, dst.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				float value = input(row * ksize.hei, col * ksize.wid, z);
				Point p(row * ksize.hei, col * ksize.wid);
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
							p.x = i;
							p.y = j;
						}
					}
				}
				markpoint(row*dst.cols() + col, 0, z) = (float)p.x;
				markpoint(row*dst.cols() + col, 1, z) = (float)p.y;
				dst(row, col, z) = value;
			}
	return dst;
}
const Mat nn::iMaxPool(const Mat & input, const Mat & markpoint, Size ksize, int stride)
{
	Mat dst = zeros(input.rows() * ksize.hei, input.cols() * ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++) {
		int row = 0;
		for (int i = 0; i < input.rows(); i++)
			for (int j = 0; j < input.cols(); j++) {
				dst((int)markpoint(row, 0, z), (int)markpoint(row, 1, z), z) = input(i, j, z);
				row += 1;
			}
	}
	return dst;
}
const Mat nn::iAveragePool(const Mat & input, Size ksize, int stride)
{
	Mat dst = zeros(input.rows() * ksize.hei, input.cols() * ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < input.rows(); row++)
			for (int col = 0; col < input.rows(); col++) {
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						dst(i, j, z) = input(row, col, z);
					}
				}
			}
	return dst;
}
const Mat nn::AveragePool(const Mat & input, Size ksize, int stride)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.rows(); col++) {
				float value = 0;
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						value += input(i, j, z);
					}
				}
				dst(row, col, z) = value / float(ksize.hei*ksize.wid);
			}
	return dst;
}
const Mat nn::FullConnection(const Mat & input, const Mat & layer, const Mat & bias)
{
	return layer * input + bias;
}

NetNode<Layer> nn::CreateNode(const Layer & data, NetNode<Layer>* parent)
{
	NetNode<Layer> netnode(data, parent);
	if(parent)
	   netnode.data.layer_index = parent->data.layer_index;
	return netnode;
}
void nn::save_layer(const NetNode<Layer>* tree, json *j, FILE *data)
{
	for (const NetNode<Layer> & node : tree->child)
	{	
		json info;
		const Layer* layer = &node.data;
		info["type"] = Layer::Type2String(layer->type);
		info["name"] = layer->name;
		info["matrix"] = false;
		info["layer"] = layer->layer_index;
		info["last"] = layer->last;
		switch (layer->type)
		{
		case nn::NONE:
			break;
		case nn::CONV2D:
			info["matrix"] = true;
			info["isact"] = layer->convInfo.isact;
			if (layer->convInfo.isact)
				info["activate"] = Func2String(layer->active.f);
			info["channel"] = layer->convInfo.channel;
			info["kernSize"] = layer->convInfo.kern_size;
			info["strides"]["height"] = layer->convInfo.strides.hei;
			info["strides"]["width"] = layer->convInfo.strides.wid;
			info["anchor"]["x"] = layer->convInfo.anchor.x;
			info["anchor"]["y"] = layer->convInfo.anchor.y;
			info["is_copy_border"] = layer->convInfo.is_copy_border;
			break;
		case nn::MAX_POOL:
			info["strides"] = layer->pInfo.strides;
			info["size"]["height"] = layer->pInfo.size.hei;
			info["size"]["width"] = layer->pInfo.size.wid;
			break;
		case nn::AVERAGE_POOL:
			info["strides"] = layer->pInfo.strides;
			info["size"]["height"] = layer->pInfo.size.hei;
			info["size"]["width"] = layer->pInfo.size.wid;
			break;
		case nn::FULLCONNECTION:
			info["matrix"] = true;
			if (layer->fcInfo.isact)
				info["activate"] = Func2String(layer->active.f);
			info["size"] = layer->fcInfo.size;
			break;
		case nn::ACTIVATION:
			info["activate"] = Func2String(layer->active.f);
			break;
		case nn::RESHAPE:
			info["size"]["x"] = layer->reshapeInfo.size.x;
			info["size"]["y"] = layer->reshapeInfo.size.y;
			info["size"]["z"] = layer->reshapeInfo.size.z;
			break;
		case nn::DROPOUT:
			info["dropout"] = layer->dropoutInfo.dropout;
			break;
		case nn::LOSS:
			info["loss"] = Func2String(layer->loss.f);
			info["ignore_active"] = layer->loss.ignore_active;
			break;
		default:
			continue;
		}
		j->push_back(info);
		if (info["matrix"])
		{
			int param[3] = { layer->param.rows(), layer->param.cols(), layer->param.channels() };
			fwrite(param, sizeof(int) * 3, 1, data);
			fwrite(layer->param, sizeof(float)*layer->param.length(), 1, data);
			int bias[3] = { layer->bias.rows(), layer->bias.cols(), layer->bias.channels() };
			fwrite(bias, sizeof(int) * 3, 1, data);
			fwrite(layer->bias, sizeof(float)*layer->bias.length(), 1, data);
		}
		if (!node.sibling.empty()) {
			for (const NetNode<Layer> & brother : node.sibling) {
				save_layer(&brother, j, data);
			}
		}
		if (!node.child.empty()) {
			save_layer(&node, j, data);
		}
	}
}
void nn::save_param_layer(const NetNode<Layer>* tree, FILE * data)
{
	for (const NetNode<Layer> & node : tree->child) {
		const Layer *layer = &node.data;
		if (layer->type == CONV2D || layer->type == FULLCONNECTION)
		{
			int param[3] = { layer->param.rows(), layer->param.cols(), layer->param.channels() };
			fwrite(param, sizeof(int) * 3, 1, data);
			fwrite(layer->param, sizeof(float)*layer->param.length(), 1, data);
			int bias[3] = { layer->bias.rows(), layer->bias.cols(), layer->bias.channels() };
			fwrite(bias, sizeof(int) * 3, 1, data);
			fwrite(layer->bias, sizeof(float)*layer->bias.length(), 1, data);
		}
		if (!node.sibling.empty()) {
			for (const NetNode<Layer> & brother : node.sibling) {
				save_param_layer(&brother, data);
			}
		}
		if (!node.child.empty()) {
			save_param_layer(&node, data);
		}
	}
}
void nn::load_layer(NetNode<Layer>* tree, FILE * data)
{
	for (NetNode<Layer> & node : tree->child){	
		Layer *layer = &node.data;
		if (layer->type == CONV2D || layer->type == FULLCONNECTION)
		{
			int param[3]; int bias[3];
			fread(param, sizeof(int) * 3, 1, data);
			layer->param = zeros(param[0], param[1], param[2]);
			fread(layer->param, sizeof(float)*layer->param.length(), 1, data);
			fread(bias, sizeof(int) * 3, 1, data);
			layer->bias = zeros(bias[0], bias[1], bias[2]);
			fread(layer->bias, sizeof(float)*layer->bias.length(), 1, data);
		}
		if (!node.sibling.empty()) {
			for (NetNode<Layer> & brother : node.sibling) {
				load_layer(&brother, data);
			}
		}
		if (!node.child.empty()) {
			load_layer(&node, data);
		}
	}
}
int nn::insert_layer(NetNode<Layer>* tree, string name, Layer &layer, bool sibling)
{
	int success_insert = 0;
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		if (!node->sibling.empty()) {
			for (NetNode<Layer> & brother : node->sibling) {
				success_insert += insert_layer(&brother, name, layer, sibling);
			}
		}
		Layer *layerInfo = &node->data;
		if (layerInfo->name == name) {
			if (sibling) {
				node->sibling.push_back(NetNode<Layer>(Layer(), nullptr));
				node->sibling[node->sibling.size() - 1].data.layer_index = node->data.layer_index + 1;
				NetNode<Layer> netnode = CreateNode(layer, &node->sibling[node->sibling.size() - 1]);
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
void nn::update_layer(NetNode<Layer>* tree, vector<Mat>& mat, int &idx)
{
	for (NetNode<Layer> & node : tree->child) {
		Layer *layerInfo = &node.data;
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layerInfo->param += mat[idx++];
			layerInfo->bias += mat[idx++];
			break;
		case nn::FULLCONNECTION:
			layerInfo->param += mat[idx++];
			layerInfo->bias += mat[idx++];
			break;
		default:
			break;
		}
		if (!node.sibling.empty()) {
			for (NetNode<Layer> & brother : node.sibling) {
				update_layer(&brother, mat, idx);
			}
		}
		if (!node.child.empty()) {
			update_layer(&node, mat, idx);
		}
	}
}
void nn::initialize_loss(NetNode<Layer>* tree, vector<NetNode<Layer>*>* loss)
{
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		Layer *layerInfo = &node->data;
		switch (layerInfo->type)
		{
		case nn::LOSS:
			if (loss) {
				loss->push_back(&*node);
				(node - 1)->data.last = node->data.loss.ignore_active;
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
			for (NetNode<Layer> & brother : node->sibling) {
				initialize_loss(&brother, loss);
			}
		}
		if (!node->child.empty()) {
			initialize_loss(&*node, loss);
		}
	}
}
void nn::initialize_size(NetNode<Layer>* tree, vector<Size3>& input_size, int idx)
{
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end();) {
		Layer *layerInfo = &node->data;
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layerInfo->param = CreateMat(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*input_size[idx].z);
			layerInfo->bias = zeros(1, 1, layerInfo->convInfo.channel);
			if (!layerInfo->convInfo.is_copy_border)
				input_size[idx] = mCalSize(input_size[idx], Size3(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*input_size[idx].z), layerInfo->convInfo.anchor, layerInfo->convInfo.strides);
			else
				input_size[idx].z = layerInfo->convInfo.channel;
			break;
		case nn::MAX_POOL:
			input_size[idx].x /= layerInfo->pInfo.size.hei;
			input_size[idx].y /= layerInfo->pInfo.size.wid;
			break;
		case nn::AVERAGE_POOL:
			input_size[idx].x /= layerInfo->pInfo.size.hei;
			input_size[idx].y /= layerInfo->pInfo.size.wid;
			break;
		case nn::FULLCONNECTION:
			if (input_size[idx].z != 1 || input_size[idx].y != 1)
			{
				node = tree->child.insert(node, CreateNode(Reshape(Size3(input_size[idx].x*input_size[idx].y*input_size[idx].z, 1, 1)), &(*node)));
				continue;
			}
			layerInfo->param = CreateMat(layerInfo->fcInfo.size, input_size[idx].x, 1);
			layerInfo->bias = zeros(layerInfo->fcInfo.size, 1, 1);
			input_size[idx] = Size3(layerInfo->param.rows(), 1, 1);
			break;
		case nn::ACTIVATION:
			break;
		case nn::RESHAPE:
			input_size[idx] = layerInfo->reshapeInfo.size;
			break; 
		default:
			break;
		}
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
void nn::initialize_channel(NetNode<Layer>* tree, vector<int>* input_channel, int idx)
{
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		Layer *layerInfo = &node->data;
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layerInfo->param = CreateMat(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*(*input_channel)[idx]);
			layerInfo->bias = zeros(1, 1, layerInfo->convInfo.channel);
			(*input_channel)[idx] = layerInfo->convInfo.channel;
			break;
		case nn::FULLCONNECTION:
			fprintf(stderr, "任意尺寸初始化不允许有全连接层!\n");
			throw FULLCONNECTION;
		default:
			break;
		}
		if (!node->sibling.empty()) {
			for (int i = 0; i < node->sibling.size(); ++i) {
				input_channel->push_back((*input_channel)[idx]);
				initialize_channel(&node->sibling[i], input_channel, idx + 1 + i);
			}
		}
		if (!node->child.empty()) {
			initialize_channel(&*node, input_channel, idx);
		}
	}
}
void nn::initialize_mat(const NetNode<Layer>* tree, vector<Size3>* mat_size)
{
	for (const NetNode<Layer> & node : tree->child) {
		const Layer *layerInfo = &node.data;
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			mat_size->push_back(layerInfo->param.size3());
			mat_size->push_back(layerInfo->bias.size3());
			break;
		case nn::FULLCONNECTION:
			mat_size->push_back(layerInfo->param.size3());
			mat_size->push_back(layerInfo->bias.size3());
			break;
		default:
			break;
		}
		if (!node.sibling.empty()) {
			for (const NetNode<Layer> brother : node.sibling)
				initialize_mat(&brother, mat_size);
		}
		if (!node.child.empty()) {
			initialize_mat(&node, mat_size);
		}
	}
}
void nn::forward(const NetNode<Layer>* tree, vector<Mat>& output, int idx)
{
	for (const NetNode<Layer> & layerInfo : tree->child) {
		const Layer *layer = &layerInfo.data;
		switch (layer->type)
		{
		case CONV2D:
			output[idx] = conv2d(output[idx], layer->param, layer->convInfo.strides, layer->convInfo.anchor, layer->convInfo.is_copy_border) + layer->bias;
			if (layer->convInfo.isact)
				output[idx] = layer->active.f(output[idx]);
			break;
		case MAX_POOL:
			output[idx] = MaxPool(output[idx], layer->pInfo.size, layer->pInfo.strides);
			break;
		case AVERAGE_POOL:
			output[idx] = AveragePool(output[idx], layer->pInfo.size, layer->pInfo.strides);
			break;
		case FULLCONNECTION:
			output[idx] = FullConnection(output[idx], layer->param, layer->bias);
			if (layer->fcInfo.isact)
				output[idx] = layer->active.f(output[idx]);
			break;
		case ACTIVATION:
			output[idx] = layer->active.f(output[idx]);
			break;
		case RESHAPE:
			output[idx].reshape(layer->reshapeInfo.size);
			break;
		default:
			break;
		}
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				output.push_back(output[idx]);
				forward(&layerInfo.sibling[i], output, idx + 1 + i);
			}
		}
		if (!layerInfo.child.empty()) {
			forward(&layerInfo, output, idx);
		}
	}
}
void nn::forward_train(const NetNode<Layer>* tree, vector<vector<Mat>>& variable, vector<Mat>& output, int idx)
{
	for (const NetNode<Layer> & layerInfo : tree->child) {
		const Layer *layer = &layerInfo.data;
		switch (layer->type)
		{
		case CONV2D:
			output[idx] = conv2d(output[idx], layer->param, layer->convInfo.strides, layer->convInfo.anchor, layer->convInfo.is_copy_border) + layer->bias;
			variable[idx].push_back(output[idx]);
			if (layer->convInfo.isact)
				output[idx] = layer->active.f(output[idx]);
			break;
		case MAX_POOL:
		{
			Mat mark;
			output[idx] = MaxPool(output[idx], mark, layer->pInfo.size, layer->pInfo.strides);
			variable[idx].push_back(mark);
			variable[idx].push_back(output[idx]);
		}
		break;
		case AVERAGE_POOL:
			output[idx] = AveragePool(output[idx], layer->pInfo.size, layer->pInfo.strides);
			variable[idx].push_back(output[idx]);
			break;
		case FULLCONNECTION:
			output[idx] = FullConnection(output[idx], layer->param, layer->bias);
			variable[idx].push_back(output[idx]);
			if (layer->fcInfo.isact)
				output[idx] = layer->active.f(output[idx]);
			break;
		case ACTIVATION:
			output[idx] = layer->active.f(output[idx]);
			break;
		case RESHAPE:
		{
			Mat mat(3, 1, 1);
			mat(0) = (float)output[idx].rows();
			mat(1) = (float)output[idx].cols();
			mat(2) = (float)output[idx].channels();
			variable[idx].push_back(mat);
			output[idx].reshape(layer->reshapeInfo.size);
			variable[idx].push_back(output[idx]);
		}
		break;
		case DROPOUT:
			if (layer->dropoutInfo.dropout != 0) {
				Mat drop = mThreshold(mRand(0, 1, output[idx].size3(), true), layer->dropoutInfo.dropout, 0, 1);
				output[idx] = Mult(output[idx], drop);
				output[idx] *= 1.0f / (1 - layer->dropoutInfo.dropout);
			}
			break;
		default:break;
		}
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				variable.push_back(vector<Mat>(1, variable[idx][variable[idx].size() - 1]));
				output.push_back(output[idx]);
				forward_train(&layerInfo.sibling[i], variable, output, idx + 1 + i);
			}
		}
		if (!layerInfo.child.empty()) {
			forward_train(&layerInfo, variable, output, idx);
		}
	}
}
void nn::back_train(const NetNode<Layer>* tree, vector<Mat>& dlayer, vector<vector<Mat>>& x, vector<Mat>& output, int &number, int idx)
{
	int x_num = (int)x[idx].size() - 1;
	for (int index = (int)tree->child.size() - 1; index >= 0; --index) {
		if (!tree->child[index].sibling.empty()) {
			for (int i = 0; i < tree->child[index].sibling.size(); ++i) {
				back_train(&tree->child[index].sibling[i], dlayer, x, output, number, idx + i + 1);
				output[idx] = output[idx] + output[idx + i + 1];
			}
			//output[idx] /= (float)tree->child[index].sibling.size();
		}
		const Layer *layer = &tree->child[index].data;
		switch (tree->child[index].data.type)
		{
		case nn::CONV2D:
			if (!layer->last && layer->convInfo.isact)
				output[idx] = Mult(output[idx], layer->active.df(x[idx][x_num]));
			dlayer[dlayer.size() - 1 - number] = mSum(output[idx], CHANNEL);
			dlayer[dlayer.size() - 2 - number] = iconv2d(x[idx][x_num - 1], output[idx], layer->param.size3(), layer->convInfo, false);
			if (idx != 0||index > 0)
				output[idx] = iconv2d(output[idx], rotate(tree->child[index].data.param, ROTATE_180_ANGLE), Size3(), layer->convInfo, true);
			x_num -= 1;
			number += 2;
			break;
		case nn::MAX_POOL:
			output[idx] = upsample(output[idx], layer->pInfo.size, layer->pInfo.strides, x[idx][x_num - 1]);
			x_num -= 2;
			break;
		case nn::AVERAGE_POOL:
			output[idx] = upsample(output[idx], layer->pInfo.size, layer->pInfo.strides);
			x_num -= 1;
			break;
		case nn::FULLCONNECTION:
			if (!layer->last && layer->fcInfo.isact)
				output[idx] = Mult(output[idx], layer->active.df(x[idx][x_num]));
			dlayer[dlayer.size() - 1 - number] = output[idx];
			dlayer[dlayer.size() - 2 - number] = output[idx] * x[idx][x_num - 1].t();
			number += 2;
			if (idx != 0 || index > 0)
				output[idx] = tree->child[index].data.param.t() * output[idx];
			x_num -= 1;
			break;
		case nn::ACTIVATION:
			if (!layer->last)
				output[idx] = Mult(output[idx], layer->active.df(x[idx][x_num]));
			break;
		case nn::RESHAPE:
			//if (x[idx][x_num].length() != 3)x_num -= 1;
			output[idx].reshape((int)x[idx][x_num - 1](0), (int)x[idx][x_num - 1](1), (int)x[idx][x_num - 1](2));
			x_num -= 2;
			break;
		default:
			break;
		}
		if (!tree->child[index].child.empty()) {
			back_train(&tree->child[index], dlayer, x, output, number, idx);
		}
	}
}
void nn::show_net(const NetNode<Layer>* tree, std::ostream & out)
{
	for (const NetNode<Layer> & layerInfo : tree->child) {
		Layer layer = layerInfo.data;
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

void nn::generateBbox(const Mat &score, const Mat &location, std::vector<Bbox> &boundingBox_, float scale, float threshold)
{
	const int stride = 2;
	const int cellsize = 12;
	Bbox bbox{};
	const float thres = threshold;
	float inv_scale = 1.0f / scale;
	for (int row = 0; row < score.rows(); row++) {
		bbox.y1 = (int)((stride * row + 1) * inv_scale + 0.5f);
		bbox.y2 = (int)((stride * row + 1 + cellsize) * inv_scale + 0.5f);
		int diff_y = (bbox.y2 - bbox.y1);
		for (int col = 0; col < score.cols(); col++) {
			if (score(row, col, 1) > thres) {
				bbox.score = score(row, col, 1);
				bbox.x1 = (int)((stride * col + 1) * inv_scale + 0.5f);
				bbox.x2 = (int)((stride * col + 1 + cellsize) * inv_scale + 0.5f);
				bbox.area = (float)(bbox.x2 - bbox.x1) * diff_y;
				for (int channel = 0; channel < 4; channel++) {
					bbox.regreCoord[channel] = location(row, col, channel);
				}
				boundingBox_.push_back(bbox);
			}
		}
	}
}

void nn::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname) 
{
	if (boundingBox_.empty()) {
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	std::vector<int> vPick;
	int nPick = 0;
	std::multimap<float, int> vScores;
	const int num_boxes = (int)boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(std::pair<float, int>(boundingBox_[i].score, i));
	}
	while (vScores.size() > 0) {
		int last = vScores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;
		for (std::multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();) {
			int it_idx = it->second;
			maxX = (float)std::max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
			maxY = (float)std::max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
			minX = (float)std::min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
			minY = (float)std::min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
			//maxX1 and maxY1 reuse 
			maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (!modelname.compare("Union"))
				IOU = IOU / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - IOU);
			else if (!modelname.compare("Min")) {
				IOU = IOU / ((boundingBox_.at(it_idx).area < boundingBox_.at(last).area) ? boundingBox_.at(it_idx).area
					: boundingBox_.at(last).area);
			}
			if (IOU > overlap_threshold) {
				it = vScores.erase(it);
			}
			else {
				it++;
			}
		}
	}

	vPick.resize(nPick);
	std::vector<Bbox> tmp_;
	tmp_.resize(nPick);
	for (int i = 0; i < nPick; i++) {
		tmp_[i] = boundingBox_[vPick[i]];
	}
	boundingBox_ = tmp_;
}

void nn::refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square) {
	if (vecBbox.empty()) {
		std::cout << "Bbox is empty!!" << std::endl;
		return;
	}
	float bbw = 0, bbh = 0, maxSide = 0;
	float h = 0, w = 0;
	float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	for (vector<Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++) {
		bbw = float((*it).x2 - (*it).x1 + 1);
		bbh = float((*it).y2 - (*it).y1 + 1);
		x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
		y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
		x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
		y2 = (*it).y2 + (*it).regreCoord[3] * bbh;


		if (square) {
			w = x2 - x1 + 1;
			h = y2 - y1 + 1;
			maxSide = (h > w) ? h : w;
			x1 = x1 + w * 0.5f - maxSide * 0.5f;
			y1 = y1 + h * 0.5f - maxSide * 0.5f;
			(*it).x2 = (int)(x1 + maxSide - 1 + 0.5f);
			(*it).y2 = (int)(y1 + maxSide - 1 + 0.5f);
			(*it).x1 = (int)(x1 + 0.5f);
			(*it).y1 = (int)(y1 + 0.5f);
		}

		//boundary check
		if ((*it).x1 < 0)(*it).x1 = 0;
		if ((*it).y1 < 0)(*it).y1 = 0;
		if ((*it).x2 > width)(*it).x2 = width - 1;
		if ((*it).y2 > height)(*it).y2 = height - 1;

		it->area = float((it->x2 - it->x1) * (it->y2 - it->y1));
	}
}

void nn::resize(const Mat & src, Mat & dst, float xRatio, float yRatio, ReductionMothed mothed)
{
	if (src.empty())return;
	int rows = static_cast<int>(src.rows() * yRatio);
	int cols = static_cast<int>(src.cols() * xRatio);
	Mat img(rows, cols, src.channels());
	switch (mothed)
	{
	case nn::EqualIntervalSampling:
		for (int i = 0; i < rows; i++) {
			int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
			for (int j = 0; j < cols; j++) {
				int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
				img(i, j) = src(row, col); //取得采样像素
			}
		}
		break;
	case nn::LocalMean:
	{
		int lastRow = 0;
		int lastCol = 0;

		for (int i = 0; i < rows; i++) {
			int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
			for (int j = 0; j < cols; j++) {
				int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
				Vec<float> temp;
				for (int idx = lastCol; idx <= col; idx++) {
					for (int jdx = lastRow; jdx <= row; jdx++) {
						temp[0] += src(jdx, idx, 0);
						temp[1] += src(jdx, idx, 1);
						temp[2] += src(jdx, idx, 2);
					}
				}

				int count = (col - lastCol + 1) * (row - lastRow + 1);
				img(i, j, 0) = temp[0] / count;
				img(i, j, 1) = temp[1] / count;
				img(i, j, 2) = temp[2] / count;

				lastCol = col + 1; //下一个子块左上角的列坐标，行坐标不变
			}
			lastCol = 0; //子块的左上角列坐标，从0开始
			lastRow = row + 1; //子块的左上角行坐标
		}
	}
	break;
	default:
		break;
	}
	dst = img;
}

void nn::resize(const Mat & src, Mat & dst, Size newSize, ReductionMothed mothed)
{
	resize(src, dst, newSize.wid / float(src.cols()), newSize.hei / float(src.rows()), mothed);
}
