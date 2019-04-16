#include "filetools.h"
#include "TrainData.h"
#include "imgprocess.h"
#include <fstream>
using std::ifstream;
using namespace nn;

nn::TrainData::TrainData()
{
}

nn::TrainData::TrainData(string rootpath, string imglist, string imagedir, 
	int batch_size, const Mat(*label_proces)(const Mat&))
	: batch_size(batch_size), rootpath(rootpath)
{
	load_train_data(rootpath, imglist, imagedir, batch_size, label_proces);
}

void nn::TrainData::load_train_data(string rootpath, string imglist, string imagedir,
	int batch_size, const Mat(*label_proces)(const Mat&))
{
	clear();
	this->rootpath = rootpath;
	this->batch_size = batch_size;
	get_train_data(imglist, imagedir, label_proces);
	range.resize(imgpath.size());
	for (size_t i = 0; i < imgpath.size(); ++i)
		range[i] = int(i);
}


nn::TrainData::~TrainData()
{
}

void nn::TrainData::reset()
{
	for (int &v : range)
	{
		int idx;
		do {
			idx = rand() % range.size();
		} while (range[idx] == v);
		int temp = range[idx];
		range[idx] = v;
		v = temp;
	}
	index = 0;
}

void nn::TrainData::clear()
{
	vector<string>().swap(imgpath);
	vector<Mat>().swap(data);
	vector<Mat>().swap(label);
	vector<int>().swap(range);
}

int nn::TrainData::len() const
{
	return (int)range.size();
}

void nn::TrainData::batch(Mat & x, Mat & y)
{
	if (data.size() != range.size())
		next();
	x = data[index];
	y = label[index];
	index = (index + 1) % range.size();
}

void nn::TrainData::batches(vector<Mat>& x, vector<Mat>& y)
{
}

void nn::TrainData::loadAllData()
{
	index = 0;
	for (int &v : range) {
		next();
		index += 1;
	}
}

void nn::TrainData::register_process(void(*image)(const Image&, Image&), void(*mat)(const Mat&, Mat&))
{
	process_image = image;
	process_mat = mat;
} 

void nn::TrainData::next()
{
	Image img = Imread(imgpath[index]);
	if (process_image != nullptr)process_image(img, img);
	Mat mat = Image2Mat(img);
	if (process_mat != nullptr)process_mat(mat, mat);
	data.push_back(mat);
}

void nn::TrainData::get_train_data(string imglist, string imagepath, const Mat(*label_proces)(const Mat&))
{
	vector<string>().swap(imgpath);
	ifstream in(rootpath + "\\" + imglist);
	if (in.is_open()) {
		string str;
		while (std::getline(in, str)) {
			vector<string> v = strsplit(str, '|');
			imgpath.push_back(rootpath + "\\" + imagepath + "\\" + v[0]);
			v = strsplit(v[1], ' ');
			Mat label_(str2double(v), ROW);
			if (label_proces != nullptr)
				label.push_back(label_proces(label_));
			else
				label.push_back(label_);
		}
		in.close();
	}
}
