#include "pool.h"
#include "method.h"


nn::pool::pool(string name)
	: Layer(name)
{
	isinit = true;
	isforword = true;
	isback = true;
	isupdate = false;
	isparam = false;
	type = POOL;
}

nn::pool::~pool()
{
}

void nn::pool::forword(const Mat & in, Mat &out) const
{
	if (pool_type == max_pool)
	{
		out = MaxPool(in);
	}
	else if (pool_type == average_pool) {
		out = AveragePool(in);
	}
}

void nn::pool::forword_train(const vector<Mat> &in, vector<Mat> & out, vector<Mat> & variable)
{
	if (pool_type == max_pool)
	{
		if (markpoint.empty()) {
			markpoint.resize(in.size());
			Size3 input_size = in[0].size3();
			for (size_t idx = 0; idx < in.size(); ++idx) {
				markpoint[idx].create(input_size.h*input_size.w, 2, input_size.c);
				out[idx] = MaxPool(in[idx], idx);
			}
		}
		else {
			for (size_t idx = 0; idx < in.size(); ++idx)
				out[idx] = MaxPool(in[idx], idx);
		}
	}
	else if (pool_type == average_pool) {
		for (size_t idx = 0; idx < in.size(); ++idx) {
			out[idx] = AveragePool(in[idx]);
		}
	}
	variable = out;
}

void nn::pool::back(const vector<Mat> &in, vector<Mat> & out, vector<Mat> *dlayer,int *number)const
{
	for (size_t idx = 0; idx < in.size(); ++idx)
		out[idx] = upsample(in[idx], idx);
}

nn::Size3 nn::pool::initialize(Size3 input_size)
{
	return mCalSize(input_size, Size(ksize.h, ksize.w), Point(0, 0), Size(strides, strides));
}

void nn::pool::save(json * jarray, FILE * file) const
{
	json info;
	info["type"] = Layer::Type2String(type);
	info["name"] = name;
	info["layer"] = layer_index;
	info["pool_type"] = pool::Pooltype2String(pool_type);
	info["strides"] = strides;
	info["size"]["height"] = ksize.h;
	info["size"]["width"] = ksize.w;
	jarray->push_back(info);
}

void nn::pool::load(json & info, FILE * file)
{
	pool_type = pool::String2Pooltype(info["pool_type"]);
	strides = info["strides"];
	ksize.h = info["size"]["height"];
	ksize.w = info["size"]["width"];
	type = Layer::String2Type(info["type"]);
	layer_index = info["layer"];
}

void nn::pool::show(std::ostream & out) const
{
	out << name << "{" << std::endl;
	if(pool_type == max_pool)
		out << "Layer-> MaxPool" << std::endl;
	else if(pool_type == average_pool)
		out << "Layer-> AveragePool" << std::endl;
	out << "size-> " << ksize << std::endl;
	out << "strides-> " << strides << std::endl;
	out << "}";
}

const Mat nn::pool::upsample(const Mat & input, size_t idx)const
{
	if (pool_type == average_pool)
		return iAveragePool(input);
	else if (pool_type == max_pool)
		return iMaxPool(input, idx);
	else
		return input;
}
const Mat nn::pool::MaxPool(const Mat & input)const
{
	Mat dst(input.rows() / ksize.h, input.cols() / ksize.w, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				float value = input(0, 0, z);
				for (int i = row * ksize.h; i < row*ksize.h + ksize.h; ++i) {
					for (int j = col * ksize.w; j < col*ksize.w + ksize.w; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
						}
					}
				}
				dst(row, col, z) = value;
			}
	return dst;
}
const Mat nn::pool::MaxPool(const Mat & input, size_t idx)
{
	Mat dst(input.rows() / ksize.h, input.cols() / ksize.w, input.channels());
	Point p;
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				float value = input(row * ksize.h, col * ksize.w, z);
				p.x = row * ksize.h;
				p.y = col * ksize.w;
				for (int i = row * ksize.h; i < row*ksize.h + ksize.h; ++i) {
					for (int j = col * ksize.w; j < col*ksize.w + ksize.w; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
							p.x = i;
							p.y = j;
						}
					}
				}
				markpoint[idx](row*dst.cols() + col, 0, z) = (float)p.x;
				markpoint[idx](row*dst.cols() + col, 1, z) = (float)p.y;
				dst(row, col, z) = value;
			}
	return dst;
}
const Mat nn::pool::iMaxPool(const Mat & input, size_t idx)const
{
	Mat dst = zeros(input.rows() * ksize.h, input.cols() * ksize.w, input.channels());
	for (int z = 0; z < input.channels(); z++) {
		int row = 0;
		for (int i = 0; i < input.rows(); i++)
			for (int j = 0; j < input.cols(); j++) {
				dst((int)markpoint[idx](row, 0, z), (int)markpoint[idx](row, 1, z), z) = input(i, j, z);
				row += 1;
			}
	}
	return dst;
}
const Mat nn::pool::AveragePool(const Mat & input)const
{
	Mat dst(input.rows() / ksize.h, input.cols() / ksize.w, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.rows(); col++) {
				float value = 0;
				for (int i = row * ksize.h; i < row*ksize.h + ksize.h; ++i) {
					for (int j = col * ksize.w; j < col*ksize.w + ksize.w; ++j) {
						value += input(i, j, z);
					}
				}
				dst(row, col, z) = value / float(ksize.h*ksize.w);
			}
	return dst;
}
const Mat nn::pool::iAveragePool(const Mat & input)const
{
	Mat dst = zeros(input.rows() * ksize.h, input.cols() * ksize.w, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < input.rows(); row++)
			for (int col = 0; col < input.rows(); col++) {
				for (int i = row * ksize.h; i < row*ksize.h + ksize.h; ++i) {
					for (int j = col * ksize.w; j < col*ksize.w + ksize.w; ++j) {
						dst(i, j, z) = input(row, col, z);
					}
				}
			}
	return dst;
}

nn::pool::pooltype nn::pool::String2Pooltype(string str)
{
	if (str == "max_pool")
		return max_pool;
	else if (str == "average_pool")
		return average_pool;
	else
		return max_pool;
}

string nn::pool::Pooltype2String(pooltype type)
{
	if (type == max_pool)
		return "max_pool";
	else if (type == average_pool)
		return "average_pool";
	else
		return "";
}
