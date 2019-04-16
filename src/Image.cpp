#include <memory.h>
#include <iostream>
#include "Image.h"
#include "Vriable.h"
#include "imgprocess.h"

using namespace nn;
using namespace std;

Image::Image() : data(nullptr), count(new int(0)), rows(0), cols(0), channels(0) {}

Image::Image(Size3 size)
	: data(new uchar[size.x*size.y*size.z]), count(new int(0)), rows(size.x), cols(size.y), channels(size.z)
{
	memset(data, 0, sizeof(uchar)*rows*cols*channels);
}

Image::Image(Size size, int channels)
	: data(new uchar[size.hei*size.wid*channels]), count(new int(0)), rows(size.hei), cols(size.wid), channels(channels)
{
	memset(data, 0, sizeof(uchar)*rows*cols*channels);
}

Image::Image(const char * image_path) : data(nullptr), count(new int(0)), rows(0), cols(0), channels(0)
{
	*this = Imread(image_path);
}

nn::Image::Image(int rows, int cols, int channels) : data(new uchar[rows*cols*channels]), count(new int(0)), rows(rows), cols(cols), channels(channels)
{
	memset(data,0,sizeof(uchar)*rows*cols*channels);
}

Image::Image(uchar * data, int rows, int cols, int channels) : data(data), count(new int(0)), rows(rows), cols(cols), channels(channels)
{

}

Image::Image(const Image & src): data(nullptr), count(new int(0)), rows(0), cols(0), channels(0)
{
	copy(src);
}


Image::~Image()
{
	if (*count == 0) {
		if (data != nullptr) {
			delete[]data;
			data = nullptr;
		}
		delete count;
		count = nullptr;
	}
	else {
		*count -= 1;
	}
	cols = 0;
	rows = 0;
	channels = 0;
}

int Image::reCount() const
{
	return *count;
}

int Image::length() const
{
	return rows*cols*channels;
}

bool Image::empty() const
{
	return (data == nullptr);
}

void Image::copyTo(Image & src)const
{
	src = *this;
}

const Image Image::operator+(uchar value) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw Image();
	}
	Image mark(*this);
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
		throw Image();
	}
	Image mark(*this);
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
		throw Image();
	}
	if (rows != image.rows || cols != image.cols || channels != image.channels) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw Image();
	}
	Image mark(*this);
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
		throw Image();
	}
	if (rows != image.rows || cols != image.cols || channels != image.channels) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw Image();
	}
	Image mark(*this);
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
	return Vec<uchar>(data + pos.y * cols*channels + pos.x * channels, channels);
}

Vec<uchar> Image::operator()(int row, int col) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return Vec<uchar>(data + row * cols*channels + col * channels, channels);
}

Vec<uchar> Image::operator()(int index) const
{
	if (index < 0 || index >= rows * cols*channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return Vec<uchar>(data + index * channels, channels);
}

uchar & nn::Image::operator()(Point3i pos) const
{
	if (pos.y < 0 || pos.y >= rows || pos.x < 0 || pos.x >= cols || pos.z < 0 || pos.z >= channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return data[pos.y*cols*channels + pos.x * channels + pos.z];
}

uchar & Image::operator()(int row, int col, int channel) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols || channel < 0 || channel >= channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return data[row*cols*channels + col * channels + channel];
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
		}
	}
	count = src.count;
	*count += 1;
	rows = src.rows;
	cols = src.cols;
	data = src.data;
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
					delete[] data;
					data = nullptr;
				}
			}
		}
		count = new int(0);
		rows = src.rows;
		cols = src.cols;
		channels = src.channels;
		data = new uchar[rows*cols*channels];
		memcpy(data, src.data, sizeof(uchar)*rows*cols*channels);
	}
}

const Image nn::operator+(const uchar value, const Image &src)
{
	return src + value;
}

const Image nn::operator-(const uchar value, const Image & src)
{
	return src - value;
}


