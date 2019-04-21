#include "method.h"

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
		Size3 size = mCalSize(input, kern, anchor, strides, left, right, top, bottom);
		output = zeros(size);
	}
	for (int i = 0; i < kern.channels() / input.channels(); i++) {
		Mat sum = zeros(output.rows(), output.cols());
		for (int j = 0; j < input.channels(); j++)
			sum += Filter2D(input[j], kern[i*input.channels() + j], anchor, strides, is_copy_border);
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
	markpoint = zeros(dst.rows()*dst.cols(), 2, dst.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				float value = input(0, 0, z);
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
							markpoint(row*dst.cols() + col, 0, z) = (float)i;
							markpoint(row*dst.cols() + col, 1, z) = (float)j;
						}
					}
				}
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
	return NetNode<Layer>(data, parent);
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
				node->sibling.push_back(NetNode<Layer>(Layer(), &*node));
				node->sibling[node->sibling.size() - 1].child.push_back(CreateNode(layer, &node->sibling[node->sibling.size() - 1]));
			}
			else {
				node = node->parent->child.insert(node + 1, (CreateNode(layer, &*node)));
			}
			success_insert += 1;
		}
		if (!node->child.empty()) {
			success_insert += insert_layer(&*node, name, layer, sibling);
		}
	}
	return success_insert;
}
void nn::update_layer(NetNode<Layer>* tree, vector<Mat>& mat, int idx)
{
	for (NetNode<Layer> & node : tree->child) {
		Layer *layerInfo = &node.data;
		if (!node.sibling.empty()) {
			for (NetNode<Layer> & brother : tree->sibling) {
				update_layer(&brother, mat, idx);
			}
		}
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
		if (!node.child.empty()) {
			update_layer(&node, mat, idx);
		}
	}
}
void nn::initialize_loss(NetNode<Layer>* tree, vector<NetNode<Layer>*>* loss)
{
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		if (!node->sibling.empty()) {
			for (NetNode<Layer> & brother : node->sibling) {
				initialize_loss(&brother, loss);
			}
		}
		Layer *layerInfo = &node->data;
		switch (layerInfo->type)
		{
		case nn::LOSS:
			if (loss) {
				loss->push_back(&*node);
				(node - 1)->data.last = node->data.loss.ignore_acctive;
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
		if (!node->child.empty()) {
			initialize_loss(&*node, loss);
		}
	}
}
void nn::initialize_size(NetNode<Layer>* tree, vector<Size3>& input_size, int idx)
{
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end();) {
		Layer *layerInfo = &node->data;
		if (!node->sibling.empty()) {
			if (!node->sibling.empty()) {
				for (int i = 0; i < node->sibling.size(); ++i) {
					input_size.push_back(input_size[idx]);
					initialize_size(&node->sibling[i], input_size, idx + 1 + i);
				}
			}
		}
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layerInfo->param = CreateMat(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*input_size[idx].z);
			layerInfo->bias = CreateMat(1, 1, layerInfo->convInfo.channel);
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
			layerInfo->bias = CreateMat(layerInfo->fcInfo.size, 1, 1);
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
		if (!node->child.empty()) {
			initialize_size(&*node, input_size, idx);
		}
		++node;
	}
}
void nn::initialize_channel(NetNode<Layer>* tree, vector<int>* input_channel, int idx)
{
	for (vector<NetNode<Layer>>::iterator node = tree->child.begin(); node != tree->child.end(); ++node) {
		if (!node->sibling.empty()) {
			for (int i = 0; i < node->sibling.size(); ++i) {
				input_channel->push_back((*input_channel)[idx]);
				initialize_channel(&node->sibling[i], input_channel, idx + 1 + i);
			}
		}
		Layer *layerInfo = &node->data;
		switch (layerInfo->type)
		{
		case nn::CONV2D:
			layerInfo->param = CreateMat(layerInfo->convInfo.kern_size, layerInfo->convInfo.kern_size, layerInfo->convInfo.channel*(*input_channel)[idx]);
			layerInfo->bias = CreateMat(1, 1, layerInfo->convInfo.channel);
			(*input_channel)[idx] = layerInfo->convInfo.channel;
			break;
		case nn::FULLCONNECTION:
			fprintf(stderr, "任意尺寸初始化不允许有全连接层!\n");
			throw FULLCONNECTION;
		default:
			break;
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
		if (!node.sibling.empty()) {
			for (const NetNode<Layer> brother : node.sibling)
				initialize_mat(&brother, mat_size);
		}
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
		if (!node.child.empty()) {
			initialize_mat(&node, mat_size);
		}
	}
}
void nn::forward(const NetNode<Layer>* tree, vector<Mat>& output, int idx)
{
	for (const NetNode<Layer> & layerInfo : tree->child) {
		Layer layer = layerInfo.data;
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				output.push_back(output[idx]);
				forward(&layerInfo.sibling[i], output, idx + 1 + i);
			}
		}
		switch (layer.type)
		{
		case CONV2D:
			output[idx] = conv2d(output[idx], layer.param, layer.convInfo.strides, layer.convInfo.anchor, layer.convInfo.is_copy_border) + layer.bias;
			if (layer.convInfo.isact)
				output[idx] = layer.active.f(output[idx]);
			break;
		case MAX_POOL:
			output[idx] = MaxPool(output[idx], layer.pInfo.size, layer.pInfo.strides);
			break;
		case AVERAGE_POOL:
			output[idx] = AveragePool(output[idx], layer.pInfo.size, layer.pInfo.strides);
			break;
		case FULLCONNECTION:
			output[idx] = FullConnection(output[idx], layer.param, layer.bias);
			if (layer.fcInfo.isact)
				output[idx] = layer.active.f(output[idx]);
			break;
		case ACTIVATION:
			output[idx] = layer.active.f(output[idx]);
			break;
		case RESHAPE:
			output[idx].reshape(layer.reshapeInfo.size);
			break;
		default:
			break;
		}
		if (!layerInfo.child.empty()) {
			forward(&layerInfo, output, idx);
		}
	}
}
void nn::forward_train(const NetNode<Layer>* tree, vector<vector<Mat>>& variable, vector<Mat>& output, int idx)
{
	for (const NetNode<Layer> & layerInfo : tree->child) {
		Layer layer = layerInfo.data;
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				variable.push_back(vector<Mat>(1, variable[idx][variable[idx].size() - 1]));
				output.push_back(output[idx]);
				forward_train(&layerInfo.sibling[i], variable, output, idx + 1 + i);
			}
		}
		switch (layer.type)
		{
		case CONV2D:
			output[idx] = conv2d(output[idx], layer.param, layer.convInfo.strides, layer.convInfo.anchor, layer.convInfo.is_copy_border) + layer.bias;
			variable[idx].push_back(output[idx]);
			if (layer.convInfo.isact)
				output[idx] = layer.active.f(output[idx]);
			break;
		case MAX_POOL:
		{
			Mat mark;
			output[idx] = MaxPool(output[idx], mark, layer.pInfo.size, layer.pInfo.strides);
			variable[idx].push_back(mark);
			variable[idx].push_back(output[idx]);
		}
		break;
		case AVERAGE_POOL:
			output[idx] = AveragePool(output[idx], layer.pInfo.size, layer.pInfo.strides);
			variable[idx].push_back(output[idx]);
			break;
		case FULLCONNECTION:
			output[idx] = FullConnection(output[idx], layer.param, layer.bias);
			variable[idx].push_back(output[idx]);
			if (layer.fcInfo.isact)
				output[idx] = layer.active.f(output[idx]);
			break;
		case ACTIVATION:
			output[idx] = layer.active.f(output[idx]);
			break;
		case RESHAPE:
		{
			Mat mat(3, 1, 1);
			mat(0) = (float)output[idx].rows();
			mat(1) = (float)output[idx].cols();
			mat(2) = (float)output[idx].channels();
			variable[idx].push_back(mat);
			output[idx].reshape(layer.reshapeInfo.size);
			variable[idx].push_back(output[idx]);
		}
		break;
		case DROPOUT:
			if (layer.dropoutInfo.dropout != 0) {
				Mat drop = mThreshold(mRand(0, 1, output[idx].size3(), true), layer.dropoutInfo.dropout, 0, 1);
				output[idx] = Mult(output[idx], drop);
				output[idx] *= 1.0f / (1 - layer.dropoutInfo.dropout);
			}
			break;
		default:break;
		}
		if (!layerInfo.child.empty()) {
			forward_train(&layerInfo, variable, output, idx);
		}
	}
}
void nn::back_train(const NetNode<Layer>* tree, vector<Mat>& dlayer, vector<vector<Mat>>& x, vector<Mat>& output, int idx)
{
	int number = 0;
	int x_num = (int)x[idx].size() - 1;
	for (int index = (int)tree->child.size() - 1; index >= 0; --index) {
		Layer layer = tree->child[index].data;
		switch (tree->child[index].data.type)
		{
		case nn::CONV2D:
			if (!layer.last && layer.convInfo.isact)
				output[idx] = Mult(output[idx], layer.active.df(x[idx][x_num]));
			dlayer[dlayer.size() - 1 - number] = mSum(output[idx], CHANNEL);
			dlayer[dlayer.size() - 2 - number] = iconv2d(x[idx][x_num - 1], output[idx], layer.param.size3(), layer.convInfo, false);
			if (index > 0)
				output[idx] = iconv2d(output[idx], tree->child[index].data.param.t(), Size3(), layer.convInfo, true);
			x_num -= 1;
			number += 2;
			break;
		case nn::MAX_POOL:
			output[idx] = upsample(output[idx], layer.pInfo.size, layer.pInfo.strides, x[idx][x_num - 1]);
			x_num -= 2;
			break;
		case nn::AVERAGE_POOL:
			output[idx] = upsample(output[idx], layer.pInfo.size, layer.pInfo.strides);
			x_num -= 1;
			break;
		case nn::FULLCONNECTION:
			if (!layer.last && layer.fcInfo.isact)
				output[idx] = Mult(output[idx], layer.active.df(x[idx][x_num]));
			dlayer[dlayer.size() - 1 - number] = output[idx];
			dlayer[dlayer.size() - 2 - number] = output[idx] * x[idx][x_num - 1].t();
			number += 2;
			if (index > 0)
				output[idx] = tree->child[index].data.param.t() * output[idx];
			x_num -= 1;
			break;
		case nn::ACTIVATION:
			if (!layer.last)
				output[idx] = Mult(output[idx], layer.active.df(x[idx][x_num]));
			break;
		case nn::RESHAPE:
			//if (x[idx][x_num].length() != 3)x_num -= 1;
			output[idx].reshape((int)x[idx][x_num - 1](0), (int)x[idx][x_num - 1](1), (int)x[idx][x_num - 1](2));
			x_num -= 2;
			break;
		default:
			break;
		}
		if (!tree->child[idx].child.empty()) {
			forward_train(&tree->child[idx], x, output, idx);
		}
	}
}
void nn::show_net(const NetNode<Layer>* tree, std::ostream & out)
{
	for (const NetNode<Layer> & layerInfo : tree->child) {
		Layer layer = layerInfo.data;
		if (!layerInfo.sibling.empty()) {
			for (int i = 0; i < layerInfo.sibling.size(); ++i) {
				show_net(&layerInfo.sibling[i], out);
			}
		}
		out << layer << std::endl;
		if (!layerInfo.child.empty()) {
			show_net(&layerInfo, out);
		}
	}
}