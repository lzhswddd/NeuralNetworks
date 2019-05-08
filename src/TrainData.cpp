#include "filetools.h"
#include "TrainData.h"
#include "imgprocess.h"
#include <fstream>
#include <algorithm>
using std::ifstream;
using std::pair;
using namespace nn;

nn::TrainData::TrainData()
{
}

nn::TrainData::TrainData(string rootpath, string imglist, string imagedir,
	int batch_size, const vector<Mat>(*label_process)(const Mat&))
	: batch_size(batch_size), rootpath(rootpath)
{
	load_train_data(rootpath, imglist, imagedir, batch_size, label_process);
}

void nn::TrainData::load_train_data(string rootpath, string imglist, string imagedir,
	int batch_size, const vector<Mat>(*label_process)(const Mat&))
{
	clear();
	this->rootpath = rootpath;
	this->batch_size = batch_size;
	get_train_data(imglist, imagedir, label_process);
	batch_number = (int)imgpath.size() / batch_size;
	data.resize(batch_size);
	range.resize(batch_number);
	traindata.resize(batch_number * batch_size);
	for (int i = 0; i < batch_number; ++i)
		range[i] = i;
}


nn::TrainData::~TrainData()
{
}

void nn::TrainData::reset()
{
	if (batchdata.size() == range.size()) {
		random_shuffle(traindata.begin(), traindata.end());
		index = 0;
		for (vector<const NetData*> &vec : batchdata)
		{
			for (const NetData* mat : vec)
			{
				mat = &traindata[index++];
			}
		}
	}
	index = 0;
}

void nn::TrainData::clear()
{
	vector<vector<const NetData*>>().swap(batchdata);
	vector<NetData>().swap(traindata);
	vector<int>().swap(range);
}

int nn::TrainData::batchSize() const
{
	return batch_size;
}

bool nn::TrainData::all_load() const
{
	return batchdata.size() == range.size();
}

int nn::TrainData::len() const
{
	return (int)range.size();
}

TrainData::iterator nn::TrainData::batches()
{
	vector<const NetData*>* batch;
	if (batchdata.size() == batch_number) {
		batch = &batchdata[index];
	}
	else {
		next();
		batch = &batchdata[index];
		if (batchdata.size() == batch_number) {
			vector<vector<Mat>>().swap(label);
			vector<Mat>().swap(data);
			vector<string>().swap(imgpath);
		}
	}
	index = (index + 1) % batch_number;
	return batch;
}

TrainData::Databox nn::TrainData::all_batches() const
{
	return &batchdata;
}

void nn::TrainData::load_all_data(bool is_show)
{
	index = 0;
	for (int &v : range) {
		next();
		index += 1;
		if (is_show) {
			printf("load batch %d, all batches %d\n", v + 1, batch_number);
		}
	}
	vector<vector<Mat>>().swap(label);
	vector<Mat>().swap(data);
	vector<string>().swap(imgpath);
	index = 0;
}

void nn::TrainData::register_process(void(*image)(const Image&, Image&), void(*mat)(const Mat&, Mat&))
{
	process_image = image;
	process_mat = mat;
}

void nn::TrainData::next()
{
	vector<const NetData*> netData(batch_size);
	for (int i = 0; i < batch_size; ++i) {
		Image img = Imread(imgpath[index*batch_size + i]);
		if (process_image != nullptr)process_image(img, img);
		Mat mat = Image2Mat(img);
		if (process_mat != nullptr)process_mat(mat, mat);
		traindata[index*batch_size + i].input = mat;
		traindata[index*batch_size + i].label = label[index*batch_size + i];
		netData[i] = &traindata[index*batch_size + i];
	}
	batchdata.push_back(netData);
}

void nn::TrainData::get_train_data(string imglist, string imagepath, const vector<Mat>(*label_process)(const Mat&))
{
	vector<string>().swap(imgpath);
	ifstream in(rootpath + "\\" + imglist);
	vector<string> file;
	if (in.is_open()) {
		string str;
		while (std::getline(in, str)) {
			file.push_back(str);
		}
		in.close();
		random_shuffle(file.begin(), file.end());
		for (string &s : file) {
			vector<string> v = strsplit(s, ' ');
			imgpath.push_back(rootpath + "\\" + imagepath + "\\" + v[0]);
			v.erase(v.begin());
			Mat label_(str2float(v), ROW);
			if (label_process != nullptr)
				label.push_back(label_process(label_));
			else {
				label.push_back(labelProcess(label_));
			}
		}
	}
}

const vector<Mat> nn::TrainData::labelProcess(const Mat &l)
{
	return vector<Mat>(1, l);
}
