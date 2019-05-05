#include "method.h"
#include "json.hpp"
#include "costfunction.h"
#include "reshape.h"
#include <iostream>
#include <fstream>
using json = nlohmann::json;

using namespace nn;

const Mat nn::method::Xavier(int row, int col, int channel)
{
	Mat m(row, col, channel);
	float *p = m;
	for (int i = 0; i < m.length(); ++i)
	{
		*p = generateGaussianNoise(0, 1) * sqrt(6.0f / (float)(row*col*channel));
		//*p = generateGaussianNoise(0, 1) * sqrt(6.0f / (col + row));
		p++;
	}
	return m;
}
const Mat nn::method::Random(int row, int col, int channel)
{
	Mat m(row, col, channel);
	float *p = m;
	for (int i = 0; i < m.length(); ++i)
	{
		*p = generateGaussianNoise(0, 1) * 0.01f;
		p++;
	}
	return m;
}
const Mat nn::method::Random(int row, int col, int channel_input, int channel_output)
{
	Mat m(row, col, channel_output * channel_input);
	float *p = m;
	for (int i = 0; i < m.length(); ++i)
	{
		*p = generateGaussianNoise(0, 1) * 0.01f;
		p++;
	}
	return m;
}

NetNode<Layer*> nn::method::CreateNode(Layer* data, NetNode<Layer*>* parent)
{
	NetNode<Layer*> netnode(data, parent);
	if(parent)
	   netnode.data->layer_index = parent->data->layer_index;
	return netnode;
}
void nn::method::save_layer(const NetNode<Layer*>* tree, json *j, FILE *data)
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
void nn::method::save_param_layer(const NetNode<Layer*>* tree, FILE * data)
{
	for (const NetNode<Layer*> & node : tree->child) {
		if(node.data->isparam)
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
void nn::method::load_layer(NetNode<Layer*>* tree, FILE * data)
{
	for (NetNode<Layer*> & node : tree->child){	
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
int nn::method::insert_layer(NetNode<Layer*>* tree, string name, Layer *layer, bool sibling)
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
				node->sibling.push_back(NetNode<Layer*>(total,  nullptr));
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
void nn::method::update_layer(NetNode<Layer*>* tree, vector<Mat>& mat, int &idx)
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
void nn::method::regularization(NetNode<Layer*>* tree, float lambda)
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
void nn::method::initialize_loss(NetNode<Layer*>* tree, vector<NetNode<Layer*>*>* loss)
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
void nn::method::initialize_size(NetNode<Layer*>* tree, vector<Size3>& input_size, int idx)
{
	for (vector<NetNode<Layer*>>::iterator node = tree->child.begin(); node != tree->child.end();) {
		if (node->data->type == FULLCONNECTION)
		{
			if (input_size[idx].c != 1 || input_size[idx].w != 1)
			{
				node = tree->child.insert(node, CreateNode(Reshape(Size3(input_size[idx].h*input_size[idx].w*input_size[idx].c, 1, 1)), &(*node)));
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
void nn::method::initialize_mat(const NetNode<Layer*>* tree, vector<Size3>* mat_size)
{
	for (const NetNode<Layer*> & node : tree->child) {
		if(node.data->isparam)
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
void nn::method::delete_layer(NetNode<Layer*>* tree)
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
void nn::method::forward(const NetNode<Layer*>* tree, vector<Mat>& output, int idx)
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
void nn::method::forward_train(NetNode<Layer*>* tree, vector<Mat> & variable, vector<vector<Mat>>& output, int idx)
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
void nn::method::back_train(NetNode<Layer*>* tree, vector<Mat>& dlayer, vector<vector<Mat>> & output, int &number, int idx)
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
		if (layer->isback) {
			layer->back(output[idx], output[idx], &dlayer, &number);
		}
		if (!tree->child[index].child.empty()) {
			back_train(&tree->child[index], dlayer, output, number, idx);
		}
	}
}
void nn::method::save_mat(FILE * file, const Mat & m)
{
	int paramsize[3] = { m.rows(), m.cols(), m.channels() };
	fwrite(paramsize, sizeof(int) * 3, 1, file);
	fwrite(m, sizeof(float)*m.length(), 1, file);
}
void nn::method::load_mat(FILE * file, Mat & m)
{
	int paramsize[3];
	fread(paramsize, sizeof(int) * 3, 1, file);
	m = zeros(paramsize[0], paramsize[1], paramsize[2]);
	fread(m, sizeof(float)*m.length(), 1, file);
}
void nn::method::show_net(const NetNode<Layer*>* tree, std::ostream & out)
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

void nn::method::generateBbox(const Mat &score, const Mat &location, std::vector<Bbox> &boundingBox_, float scale, float threshold)
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

void nn::method::nms(std::vector<Bbox> &boundingBox_, const float overlap_threshold, string modelname) 
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

void nn::method::refine(vector<Bbox> &vecBbox, const int &height, const int &width, bool square) {
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

void nn::method::resize(const Mat & src, Mat & dst, float xRatio, float yRatio, ReductionMothed mothed)
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
				for (int z = 0; z < src.channels(); z++) {
					img(i, j, z) = src(row, col, z); //取得采样像素
				}
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

void nn::method::resize(const Mat & src, Mat & dst, Size newSize, ReductionMothed mothed)
{
	resize(src, dst, newSize.w / float(src.cols()), newSize.h / float(src.rows()), mothed);
}

float nn::method::generateGaussianNoise(float mu, float sigma)
{
	const float epsilon = std::numeric_limits<float>::min();
	const float two_pi = 2.0*3.14159265358979323846f;

	static float z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
		return z1 * sigma + mu;

	float u1, u2;
	do
	{
		u1 = rand() * (1.0f / RAND_MAX);
		u2 = rand() * (1.0f / RAND_MAX);
	} while (u1 <= epsilon);

	z0 = sqrt(-2.0f * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0f * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}
