#include <memory.h>
#include <iostream>
#include "alignMalloc.h"
#include "image.h"
#include "vriable.h"
#include "imgprocess.h"

#ifndef MIN
#define MIN(x, y) ((x)>(y)?(y):(x))
#endif
#ifndef MAX
#define MAX(x, y) ((x)<(y)?(y):(x))
#endif
using namespace nn;
using namespace std;

Image::Image() : data(nullptr), count(nullptr), rows(0), cols(0), channels(0) {}

Image::Image(Size3 size)
	: data((uchar*)fastMalloc(size.h*size.w*size.c * sizeof(uchar))), count(new int(0)), rows(size.h), cols(size.w), channels(size.c)
{
	step = cols;
	memset(data, 0, sizeof(uchar)*rows*cols*channels);
}

Image::Image(Size size, int channels)
	: data((uchar*)fastMalloc(size.h*size.w*channels * sizeof(uchar))), count(new int(0)), rows(size.h), cols(size.w), channels(channels)
{
	step = cols;
	memset(data, 0, sizeof(uchar)*rows*cols*channels);
}

Image::Image(string image_path, bool is_gray) : data(nullptr), count(nullptr), rows(0), cols(0), channels(0)
{
	*this = Imread(image_path, is_gray);
}

nn::Image::Image(int rows, int cols, int channels) : data((uchar*)fastMalloc(rows*cols*channels * sizeof(uchar))), count(new int(0)), rows(rows), cols(cols), channels(channels)
{
	step = cols;
	memset(data,0,sizeof(uchar)*rows*cols*channels);
}

Image::Image(uchar * data, int rows, int cols, int channels, int step, bool iscopy)
	: rows(rows), cols(cols), channels(channels), step(cols)
{
	if (iscopy) {
		this->data = (uchar*)fastMalloc(rows*cols*channels * sizeof(uchar));
		for (int row = 0; row < rows; ++row) {
			memcpy(this->data + row * cols*channels, data + row * step*channels, cols*channels * sizeof(uchar));
		}
		count = new int(0);
	}
	else {
		count = nullptr;
	}
}

Image::Image(const Image & src): data(nullptr), count(new int(0)), rows(0), cols(0), channels(0), step(0)
{
	*this = src;
}


Image::~Image()
{
	release();
}

int Image::reCount() const
{
	return count == 0 ? -1 : *count;
}

int Image::length() const
{
	return rows*cols*channels;
}

bool Image::empty() const
{
	return (data == nullptr);
}

void Image::create(int rows, int cols)
{
	*this = Image(rows, cols);
}

void Image::create(int rows, int cols, int channels)
{
	*this = Image(rows, cols, channels);
}

void Image::copyTo(Image & src)const
{
	src.copy(*this);
}

void Image::release()
{
	if (count != 0) {
		if (*count == 0) {
			if (data != nullptr) {
				fastFree(data);
				data = nullptr;
			}
			delete count;
			count = nullptr;
		}
		else {
			*count -= 1;
		}
	}
	step = 0;
	cols = 0;
	rows = 0;
	channels = 0;
}

const Image Image::operator+(uchar value) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	Image mark;
	copyTo(mark);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) + value;
	return mark;
}

const Image Image::operator-(uchar value) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	Image mark;
	copyTo(mark);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) - value;
	return mark;
}

const Image Image::operator+(const Image & image) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (rows != image.rows || cols != image.cols || channels != image.channels) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
	Image mark;
	copyTo(mark);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) + image(i, j, k);
	return mark;
}

const Image Image::operator-(const Image & image) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (rows != image.rows || cols != image.cols || channels != image.channels) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
	Image mark;
	copyTo(mark);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) - image(i, j, k);
	return mark;
}

void Image::operator+=(const Image & image)
{
	*this = *this + image;
}

void Image::operator-=(const Image & image)
{
	*this = *this - image;
}

void Image::operator+=(uchar value)
{
	*this = *this + value;
}

void Image::operator-=(uchar value)
{
	*this = *this - value;
}

Vec<uchar> Image::operator()(Point pos) const
{
	if (pos.y < 0 || pos.y >= rows || pos.x < 0 || pos.x >= cols) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return Vec<uchar>(data + pos.y * step*channels + pos.x * channels, channels);
}

Vec<uchar> Image::operator()(int row, int col) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return Vec<uchar>(data + row * step*channels + col * channels, channels);
}

Vec<uchar> Image::operator()(int index) const
{
	if (index < 0 || index >= rows * cols*channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	int row = index / cols;
	int col = index % cols;
	return Vec<uchar>(data + row * step*channels + col * channels, channels);
}

uchar & nn::Image::operator()(Point3i pos) const
{
	if (pos.y < 0 || pos.y >= rows || pos.x < 0 || pos.x >= cols || pos.z < 0 || pos.z >= channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return data[pos.y*step*channels + pos.x * channels + pos.z];
}

uchar & Image::operator()(int row, int col, int channel) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols || channel < 0 || channel >= channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return data[row*step*channels + col * channels + channel];
}

void Image::operator = (const Image & src)
{
	if (count != nullptr) {
		if (*count != 0) {
			*count -= 1;
		}
		else {
			delete count;
			count = nullptr;
			if (data != nullptr) {
				fastFree(data);
				data = nullptr;
			}
		}
	}
	count = src.count;
	*count += 1;
	step = src.step;
	rows = src.rows;
	cols = src.cols;
	channels = src.channels;
	data = src.data;
}

Image Image::ROI(Rect rect)
{
	Image image;
	image.count = count;
	*count += 1; 
	rect.y = MIN(MAX(rect.y, 0), rows - 1);
	rect.x = MIN(MAX(rect.x, 0), cols - 1);
	image.cols = MIN(MAX(rect.width, 0), cols - rect.x);
	image.rows = MIN(MAX(rect.height, 0), rows - rect.y);
	image.channels = channels;
	image.step = cols;
	image.data = data + rect.y*channels*step + rect.x*channels;
	return image;
}

Image Image::clone() const
{
	return Image(data, rows, cols, channels, step, true);
}

Image Image::operator()(Rect rect) 
{
	return ROI(rect);
}

void Image::copy(const Image & src)
{
	if (src.data != nullptr) {
		if (count != nullptr) {
			if (*count != 0) {
				*count -= 1;
				count = nullptr;
			}
			else {
				delete count;
				count = nullptr;
				if (data != nullptr) {
					fastFree(data);
					data = nullptr;
				}
			}
		}
		count = new int(0);
		step = src.step;
		rows = src.rows;
		cols = src.cols;
		channels = src.channels;
		data = (uchar*)fastMalloc(rows*cols*channels*sizeof(uchar));
		memcpy(data, src.data, sizeof(uchar)*rows*cols*channels);
	}
}

const Image nn::operator+(const uchar value, const Image &src)
{
	return src + value;
}

const Image nn::operator-(const uchar value, const Image & src)
{
	Image mark(Size3(src.rows, src.cols, src.channels));
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			for (int k = 0; k < src.channels; k++)
				mark(i, j, k) = value - src(i, j, k);
	return mark;
}


