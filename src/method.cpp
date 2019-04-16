#include "method.h"

const Mat nn::CreateMat(int row, int col, int channel, double low, double top)
{
	return mRand(0, int(top - low), row, col, channel, true) + low;
}
const Mat nn::CreateMat(int row, int col, int channel_input, int channel_output, double low, double top)
{
	return mRand(0, int(top - low), row, col, channel_output * channel_input, true) + low;
}
const Mat nn::conv2d(const Mat & input, const Mat & kern, const Size & strides, Point anchor, bool is_copy_border)
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
const Mat nn::MaxPool(const Mat & input, const Size & ksize, int stride)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				double value = 0;
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
const Mat nn::AveragePool(const Mat & input, const Size & ksize, int stride)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.rows(); col++) {
				double value = 0;
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						value += input(i, j, z);
					}
				}
				dst(row, col, z) = value / double(ksize.hei*ksize.wid);
			}
	return dst;
}
const Mat nn::FullConnection(const Mat & input, const Mat & layer, const Mat & bias)
{
	return layer * input + bias;
}
