#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip>
#include "mat.h"

using namespace std;
using namespace nn;

void nn::check(int row, int col, int depth)
{
	if (row <= 0 || col <= 0 || depth <= 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
}

void nn::Srandom()
{
	srand(uint(time(NULL)));
}

float nn::Max(const Mat &temp, bool isAbs)
{
	if (isAbs) 
		return mAbs(temp).findmax();
	else
		return temp.findmax();
}
float nn::Min(const Mat &temp, bool isAbs)
{
	if (isAbs) 
		return mAbs(temp).findmin();
	else
		return temp.findmin();
}
float nn::trace(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.isEnable() == -1){
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if(temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw std::exception(errinfo[ERR_INFO_SQUARE]);
	}
#endif //MAT_DEBUG
	float sum = 0;
	for (int index = 0; index < temp.rows(); index++) {
		sum += temp((index + index * temp.cols())*temp.channels());
	}
	return sum;
}
float nn::cof(const Mat &temp, int x, int y)
{
#ifdef MAT_DEBUG
	if (temp.isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (x >= temp.cols() || y >= temp.rows()) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
#endif //MAT_DEBUG
	temp.DimCheck();
	Mat a(temp.rows() - 1, temp.cols() - 1);
	int n = temp.rows();
	for (int i = 0, k = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if ((i != x) && (j != y)) {
				a(k) = temp(i*n + j);
				k++;
			}
	return det(a);
}
float nn::det(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	temp.DimCheck();
	if (temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw std::exception(errinfo[ERR_INFO_SQUARE]);
	}
#endif //MAT_DEBUG
	int n = temp.rows();
	if (n == 1)
		return temp(0);
	else {
		Mat a = temp.clone();
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++) {
				if (a(j + j * n) == 0) {
					float m;
					for (int d = j + 1; d < n; d++)
						if (a(j + d * n) != 0) {
							for (int f = j; f < n; f++)
								a(f + j * n) += a(f + d * n);
							m = -a(j + d * n) / a(j + j * n);
							for (int f = j; f < n; f++)
								a(f + d * n) += a(f + j * n) * m;
						}
				}
				else if (i != j) {
					float w = -a(j + i * n) / a(j + j * n);
					for (int f = j; f < n; f++)
						a(f + i * n) += a(f + j * n) * w;
				}
			}
		float answer = 1;
		for (int i = 0; i < n; i++)
			answer *= a(i + i * n);
		return answer;
	}
}
float nn::getRandData(int min, int max, bool isdouble)
{
#ifdef MAT_DEBUG
	if (min > max) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
#endif //MAT_DEBUG
	if (isdouble) {
		float m1 = (float)(rand() % 101) / 101;
		min++;
		float m2 = (float)((rand() % (max - min + 1)) + min);
		m2 = m2 - 1;
		return m1 + m2;
	}
	else {
		int m = rand() % (max - min) + 1 + min;
		return (float)m;
	}
}

float nn::mNorm(const Mat &temp, int num)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (temp.cols() != 1 && temp.rows() != 1) {
		cerr << errinfo[ERR_INFO_NORM] << endl;
		throw std::exception(errinfo[ERR_INFO_NORM]);
	}
	temp.DimCheck();
	if (num < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
#endif //MAT_DEBUG
	if (num == 1)
		return temp.sum(1, true);
	else if (num == 2)
		return sqrt(temp.sum(2, true));
	//else if (isinf(num) == 1)
	//	return abs(matrix[find(findmax())]);
	//else if (isinf(num) == -1)
	//	return abs(matrix[find(findmin())]);
	else
		return pow(temp.sum(num, true), 1 / float(num));
}
float nn::mDistance(const Mat &a, const Mat &b, int num)
{
	return (a - b).Norm(num);
}

float nn::mRandSample(const Mat &m)
{
	int row = rand() % m.rows();
	int col = rand() % m.cols();
	int depth = rand() % m.channels();
	return m(row, col, depth);
}

const Mat nn::VectoMat(vector<float> &p)
{
	Mat error;
	if (p.empty())return error;
	Mat m(int(p.size()), 1);
	for (int iter = 0; iter != int(p.size()); ++iter) {
		m(iter) = p[iter];
	}
	return m;
}
const Mat nn::VectoMat(vector<vector<float>> &ps)
{
	Mat error;
	if (ps.empty())return error;
	int size = 0;
	for (int i = 0; i < int(ps.size() - 1); ++i) {
		for (int j = i + 1; j < int(ps.size()); ++j) {
			if (ps[i].size() != ps[j].size())
				return error;
		}
	}
	int hei = int(ps.size());
	int wid = int(ps[0].size());
	Mat m(hei, wid);
	for (int i = 0; i < hei; ++i) {
		for (int j = 0; j < wid; ++j) {
			m(i, j) = ps[i][j];
		}
	}
	return m;
}
vector<float> nn::MattoVec(const Mat & m)
{
	if (m.empty())return vector<float>();
	vector<float> p(m.size());
	for (int iter = 0; iter != m.size(); ++iter) {
		p[iter] = m(iter);
	}
	return p;
}

vector<vector<float>> nn::MattoVecs(const Mat & m)
{
	if (m.empty())return vector<vector<float>>();
	vector<vector<float>> ps;
	for (int row = 0; row != m.rows(); ++row) {
		vector<float> p;
		for (int col = 0; col != m.cols(); ++col) {
			p.push_back(m(row, col));
		}
		ps.push_back(p);
	}
	return ps;  
}

const Mat nn::eye(int n)
{
	check(n, n);
	Mat mark(n, n);
	for (int ind = 0; ind < n; ind++)
		mark(ind + ind * n) = 1;
	return mark;
}
const Mat nn::mSplit(const Mat & src, int channel)
{
#ifdef MAT_DEBUG
	check(src.rows(), src.cols(), src.channels());
	if (channel > src.channels() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (channel < 0) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat mat(src.rows(), src.cols());
	for (int i = 0; i < src.rows(); i++)
		for (int j = 0; j < src.cols(); j++) 
			mat(i, j) = src(i, j, channel);
	return mat;
}
void nn::mSplit(const Mat & src, Mat * dst)
{
#ifdef MAT_DEBUG
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	for (int channel = 0; channel < src.channels(); ++channel)
		dst[channel] = src[channel];
}
const Mat nn::mMerge(const Mat * src, int channels)
{
#ifdef MAT_DEBUG
	if (channels < 0) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (src == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (src[channels - 1].empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat mat(src[0].rows(), src[0].cols(), channels);
	for (int z = 0; z < channels; z++) {
		for (int i = 0; i <src[z].rows(); i++)
			for (int j = 0; j < src[z].cols(); j++) {
				mat(i, j, z) = src[z](i, j);
			}
	}
	return mat;
}
const Mat nn::zeros(int row, int col)
{
	check(row, col);
	Mat mat(row, col);
	memset(mat, 0, sizeof(float)*mat.length());
	return mat;
}
const Mat nn::zeros(int row, int col, int channel)
{
	check(row, col, channel);
	Mat mat(row, col, channel);
	memset(mat, 0, sizeof(float)*mat.length());
	return mat;
}
const Mat nn::zeros(Size size)
{
	check(size.h, size.w);
	Mat mat(size.h, size.w);
	memset(mat, 0, sizeof(float)*mat.length());
	return mat;
}
const Mat nn::zeros(Size3 size)
{
	Mat mat(size);
	memset(mat, 0, sizeof(float)*mat.length());
	return mat;
}
const Mat nn::value(float v, int row, int col, int channel)
{
	check(row, col, channel);
	Mat mark(row, col, channel);
	for (int ind = 0; ind < row*col*channel; ind++)
		mark(ind) = v;
	return mark;
}
const Mat nn::ones(int row, int col)
{
	return value(1, row, col);
}
const Mat nn::ones(int row, int col, int channel)
{
	return value(1, row, col, channel);
}
const Mat nn::ones(Size3 size)
{
	return value(1, size.h, size.w, size.c);
}
const Mat nn::ones(Size size)
{
	check(size.h, size.w);
	return value(1, size.h, size.w);
}
const Mat nn::reverse(const Mat &m)
{
#ifdef MAT_DEBUG
	if (!(m.cols() == 1 || m.rows() == 1)) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (m.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat temp = m.clone();
	for (int ind = 0; ind < m.size() / 2; ind++) {
		float val = temp(ind);
		temp(ind) = temp((int)m.size() - 1 - ind);
		temp((int)m.size() - 1 - ind) = val;
	}
	return temp;
}
const Mat nn::mRandSample(const Mat &src, int row, int col, int channel)
{
#ifdef MAT_DEBUG
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	check(row, col, channel);
	Mat dst(row, col, channel);
	for (int ind = 0; ind < src.size(); ind++)
		dst(ind) = mRandSample(src);
	return dst;
}
const Mat nn::mRandSample(const Mat& m, X_Y_Z rc, int num)
{
#ifdef MAT_DEBUG
	if (m.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat dst = m(rand() % m.rows(), rc);
	for (int i = 1; i < num; i++) {
		dst = Mat(dst, m(rand() % m.rows(), rc), rc);
	}
	return dst;
}
const Mat nn::linspace(int low, int top, int len)
{
#ifdef MAT_DEBUG
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	check(len, len);
	Mat mark(1, len);
	mark(0)= (float)low;
	float gap = float(abs(low) + abs(top)) / (len - 1);;
	for (int ind = 1; ind < len; ind++)
		mark(ind) = mark(ind - 1) + gap;
	return mark;
}
const Mat nn::linspace(float low, float top, int len)
{
#ifdef MAT_DEBUG
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
#endif //MAT_DEBUG
	check(len, len);
	Mat mark(1, len);
	float gap = (top - low) / (len - 1);
	mark = low + linspace(0, len - 1, len)*gap;
	if (mark.isEnable() != -1) {
		mark(0) = low;
		mark(len - 1) = top;
	}
	return mark;
}
const Mat nn::copyMakeBorder(const Mat & src, int top, int bottom, int left, int right, BorderTypes borderType, float value)
{
	Size3 size = src.size3();
	size.h += (top + bottom);
	size.w += (left + right);
	Mat mat(size);
	switch (borderType)
	{
	case BORDER_CONSTANT: 
		for (int i = 0; i < top; i++) {
			for (int j = 0; j < mat.cols(); j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = value;
				}
			}
		}
		for (int i = 0; i < mat.rows(); i++) {
			for (int j = 0; j < left; j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = value;
				}
			}
		}
		for (int i = top + src.rows(); i < mat.rows(); i++) {
			for (int j = 0; j < mat.cols(); j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = value;
				}
			}
		}
		for (int i = 0; i < mat.rows(); i++) {
			for (int j = left + src.cols(); j < mat.cols(); j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = value;
				}
			}
		}
		break;
	case BORDER_REPLICATE:
		for (int i = 0; i < top; i++) {
			for (int j = left; j < mat.cols() - right; j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(0, j - left, z);
				}
			}
		}
		for (int i = top; i < mat.cols() - bottom; i++) {
			for (int j = 0; j < left; j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(i - top, 0, z);
				}
			}
		}
		for (int i = top + src.rows(); i < mat.rows(); i++) {
			for (int j = left; j < mat.cols() - right; j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(src.rows() - 1, j - left, z);
				}
			}
		}
		for (int i = top; i < mat.cols() - bottom; i++) {
			for (int j = left + src.cols(); j < mat.cols(); j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(i - top, src.cols() - 1, z);
				}
			}
		}
		for (int i = 0; i < top; i++) {
			for (int j = 0; j < left; j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(0, 0, z);
				}
			}
		}
		for (int i = 0; i < top; i++) {
			for (int j = mat.cols() - right; j < mat.cols(); j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(0, src.cols() - 1, z);
				}
			}
		}
		for (int i = mat.rows() - bottom; i < mat.rows(); i++) {
			for (int j = 0; j < left; j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(src.rows() - 1, 0, z);
				}
			}
		}
		for (int i = mat.rows() - bottom; i < mat.rows(); i++) {
			for (int j = mat.cols() - right; j < mat.cols(); j++) {
				for (int z = 0; z < mat.channels(); z++) {
					mat(i, j, z) = src(src.rows() - 1, src.cols() - 1, z);
				}
			}
		}
		break;
	case BORDER_REFLECT:
		break;
	case BORDER_WRAP:
		break;
	case BORDER_REFLECT_101:
		break;
	case BORDER_TRANSPARENT:
		break;
	case BORDER_ISOLATED:
		break;
	default:
		break;
	}
	for (int i = 0; i < src.rows(); i++) {
		for (int j = 0; j < src.cols(); j++) {
			for (int z = 0; z < src.channels(); z++) {
				mat(i + top, j + left, z) = src(i, j, z);
			}
		}
	}
	return mat;
}
const Mat nn::Block(const Mat&a, int Row_Start, int Row_End, int Col_Start, int Col_End, int Chennel_Start, int Chennel_End)
{
	int hei = Row_End - Row_Start + 1;
	int wid = Col_End - Col_Start + 1;
	int depth = Chennel_End - Chennel_Start + 1;
	check(hei, wid, depth);
	Mat mark(hei, wid, depth);
	int i = 0;
	for (int y = Row_Start, j = 0; y <= Row_End; y++, j++)
		for (int x = Col_Start, i = 0; x <= Col_End; x++, i++)
			for (int z = Chennel_Start, k = 0; z <= Chennel_End; z++, k++)
				mark(j, i, k) = a(y, x, z);
	return mark;
}
const Mat nn::mRand(int low, int top, int n, bool isdouble)
{
	return mRand(low, top, n, n, 1, isdouble);
}
const Mat nn::mRand(int low, int top, Size3 size, bool isdouble)
{
#ifdef MAT_DEBUG
	check(size.h, size.w, size.c);
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
#endif //MAT_DEBUG
	Mat m(size);
	for (int index = 0; index < m.length(); index++)
		m(index) = getRandData(low, top, isdouble);
	return m;
}
const Mat nn::mRand(int low, int top, int row, int col, int channel, bool isdouble)
{
#ifdef MAT_DEBUG
	check(row, col, channel);
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
#endif //MAT_DEBUG
	Mat m(row, col, channel);
	for (int index = 0; index < m.length(); index++)
		m(index) = getRandData(low, top, isdouble);
	return m;
}
const Mat nn::mcreate(int row, int col)
{
	return zeros(row, col);
}
const Mat nn::mcreate(int row, int col, int channel)
{
	return zeros(row, col, channel);
}
const Mat nn::mcreate(Size size)
{
	return zeros(size);
}
const Mat nn::mcreate(Size3 size)
{
	return zeros(size);
}
const Mat nn::adj(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	if (temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw temp;
	}
#endif //MAT_DEBUG
	int n = temp.rows();
	int depth = temp.channels();
	Mat a(n, n, depth);
	for (int z = 0; z < depth; z++) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				float m = cof(temp, i, j);
				a((i*n + j)*depth + z) = (float)pow(-1, i + j + 2)*m;
			}
	}
	return tran(a);
}
const Mat nn::inv(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	if (temp.channels() != 0) {
		cerr << errinfo[ERR_INFO_DIM] << endl;
		throw temp;
	}
#endif //MAT_DEBUG
	Mat m;
	for (int z = 0; z < temp.channels(); z++) {
		temp.swap(m);
		float answer = det(m);
		if (answer != 0 && answer == answer) {
			m = adj(m);
			int n = m.rows();
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++)
					m(i, j) = (1 / answer)*m(i, j);
		}
		else {
			cerr << errinfo[ERR_INFO_DET] << endl;
			throw temp;
		}
	}
	return m;
}
const Mat nn::pinv(const Mat &temp, direction direc)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
#endif //MAT_DEBUG
	switch (direc)
	{
	case LEFT:return (temp.t()*temp).inv()*temp.t();
	case RIGHT: {
		Mat m = temp.t();
		return nn::pinv(m, LEFT).t();
	}
	default: 
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw direc;
	}
}
const Mat nn::tran(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
#endif //MAT_DEBUG
	Mat a(temp.cols(), temp.rows(), temp.channels());
	int n = temp.rows(),
		m = temp.cols();
	for (int z = 0; z < temp.channels(); z++)
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				a(j, i, z) = temp(i, j, z);
	return a;
}
const Mat nn::mAbs(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
#endif //MAT_DEBUG
	Mat m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = fabs(temp(ind));
	return m;
}
const Mat nn::Rotate(float angle)
{
	Mat rotate_mat(2, 2);
	angle = angle * pi / 180.0f;
	rotate_mat(0) = cos(angle);
	rotate_mat(1) = -sin(angle);
	rotate_mat(2) = sin(angle);
	rotate_mat(3) = cos(angle);
	return rotate_mat;
}
const Mat nn::POW(const Mat &temp, int num)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	if (temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw temp;
	}
#endif //MAT_DEBUG
	Mat m;
	temp.swap(m);
	if (num > 0) {
		for (int i = 1; i < num; i++)
			m = m * temp;
		return m;
	}
	else if (num < 0) {
		Mat a(temp);
		temp.swap(a);
		m.setInv();
		a.setInv();
		for (int i = -1; i > num; i--)
			a = a * m;
		return a;
	}
	else
		return eye(temp.rows());

}
const Mat nn::mPow(const Mat &temp, float num)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = pow(temp(ind), num);
	return m;
}
const Mat nn::mSum(const Mat &temp, X_Y_Z r_c)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (r_c == COL) {
		Mat m = zeros(1, temp.cols(), 1);
		for (int i = 0; i < temp.cols(); i++)
			for (int z = 0; z < temp.channels(); z++)
				for (int j = 0; j < temp.rows(); j++)
					m(i) += temp(j, i, z);
		return m;
	}
	else if (r_c == ROW) {
		Mat m = zeros(temp.rows(), 1, 1);
		for (int i = 0; i < temp.rows(); i++)
			for (int z = 0; z < temp.channels(); z++)
				for (int j = 0; j < temp.cols(); j++)
					m(i) += temp(i, j, z);
		return m;
	}
	else if (r_c == CHANNEL) {
		Mat m = zeros(1, 1, temp.channels());
		for (int z = 0; z < temp.channels(); z++)
			for (int i = 0; i < temp.rows(); i++)
				for (int j = 0; j < temp.cols(); j++)
					m(z) += temp(i, j, z);
		return m;
	}
	else {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
}
const Mat nn::mExp(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++) {
		m(ind) = exp(temp(ind));
		if (m(ind) == 0)
			m(ind) = (numeric_limits<float>::min)();
	}
	return m;
}
const Mat nn::mLog(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++) 
		if (temp(ind) == 0)
			m(ind) = (numeric_limits<float>::min)();
		else
			m(ind) = log(temp(ind));
	return m;
}
const Mat nn::mSqrt(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = sqrt(temp(ind));
	return m;
}
const Mat nn::mOpp(const Mat &temp)
{
#ifdef MAT_DEBUG
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = -temp(ind);
	return m;
}
const Mat nn::Divi(const Mat &a, float val, direction dire)
{
#ifdef MAT_DEBUG
	if (a.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat mark(a.rows(), a.cols(), a.channels());
	for (int ind = 0; ind < mark.length(); ind++)
			if (dire == LEFT)
				mark(ind) = val / a(ind);
			else if (dire == RIGHT)
				mark(ind) = a(ind) / val;
	return mark;
}
const Mat nn::Divi(const Mat &a, const Mat &b, direction dire)
{
	switch (dire)
	{
	case LEFT:return a.inv()*b;
	case RIGHT:return a / b;
	default:
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
}
const Mat nn::Mult(const Mat &a, const Mat &b)
{
#ifdef MAT_DEBUG
	if (a.isEnable() == -1 || b.isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (b.rows() == 1 && b.cols() == 1 && a.channels() == b.channels()) {
		Matrix mat(a.size3());
		for (int i = 0; i < mat.rows(); i++)
			for (int j = 0; j < mat.cols(); j++)
				for (int z = 0; z < mat.channels(); z++)
					mat(i, j, z) = a(i, j, z) * b(z);
		return mat;
	}
	else if (a.rows() == 1 && a.cols() == 1 && a.channels() == b.channels())
	{
		Matrix mat(a.size3());
		for (int i = 0; i < mat.rows(); i++)
			for (int j = 0; j < mat.cols(); j++)
				for (int z = 0; z < mat.channels(); z++)
					mat(i, j, z) = b(i, j, z) * a(z);
		return mat;
	}
#ifdef MAT_DEBUG
	else if (a.rows() != b.rows() || a.cols() != b.cols() || a.channels() != b.channels()) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw std::exception(errinfo[ERR_INFO_MULT]);
	}
#endif //MAT_DEBUG
	Mat temp(a.rows(), a.cols(), a.channels());
	for (int ind = 0; ind < a.length(); ind++)
		temp(ind) = a(ind) * b(ind);
	return temp;
}
const Mat nn::Dot(const Mat & a, const Mat & b)
{
	return a * b;
}
const Mat nn::mMax(float a, const Mat &b)
{
#ifdef MAT_DEBUG
	if (b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a > b(ind) ? a : b(ind);
	return mark;
}
const Mat nn::mMax(const Mat &a, const Mat &b)
{
#ifdef MAT_DEBUG
	if (a.empty()|| b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (a.rows() != b.rows() || a.cols() != b.cols() || a.channels() != b.channels()) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw std::exception(errinfo[ERR_INFO_MULT]);
	}
#endif //MAT_DEBUG
	Mat mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a(ind) > b(ind) ? a(ind) : b(ind);
	return mark;
}
const Mat nn::mMin(float a, const Mat &b)
{
#ifdef MAT_DEBUG
	if (b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a < b(ind) ? a : b(ind);
	return mark;
}
const Mat nn::mMin(const Mat &a, const Mat &b)
{
#ifdef MAT_DEBUG
	if (a.empty() || b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (a.rows() != b.rows() || a.cols() != b.cols() || a.channels() != b.channels()) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw std::exception(errinfo[ERR_INFO_MULT]);
	}
#endif //MAT_DEBUG
	Mat mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a(ind) < b(ind) ? a(ind) : b(ind);
	return mark;
}
Size3 nn::mCalSize(Size3 src, Size3 kern, Point anchor, Size strides, int & top, int & bottom, int & left, int & right)
{
	int kern_row = kern.h;
	int kern_col = kern.w;
	top = anchor.x;
	bottom = kern_row - anchor.x - 1;
	left = anchor.y;
	right = kern_col - anchor.y - 1;
	return Size3(int(float(src.h - kern_row + 1) / float(strides.h) + 0.5f), int(float(src.w - kern_col + 1) / float(strides.w) + 0.5f), kern.c / src.c);
}
Size3 nn::mCalSize(Size3 src, Size3 kern, Point &anchor, Size strides)
{
	int kern_row = kern.h;
	int kern_col = kern.w;
	if (anchor == Point(-1, -1)) {
		anchor.x = kern_row / 2;
		anchor.y = kern_col / 2;
	}
	return Size3(int(float(src.h - kern_row + 1) / float(strides.h) + 0.5f), int(float(src.w - kern_col + 1) / float(strides.w) + 0.5f), kern.c / src.c);
}
Size3 nn::mCalSize(Size3 src, Size kern, Point & anchor, Size strides)
{
	int kern_row = kern.h;
	int kern_col = kern.w;
	if (anchor == Point(-1, -1)) {
		anchor.x = kern_row / 2;
		anchor.y = kern_col / 2;
	}
	return Size3(int(float(src.h - kern_row + 1) / float(strides.h) + 0.5f), int(float(src.w - kern_col + 1) / float(strides.w) + 0.5f), src.c);
}
const Mat nn::mThreshold(const Mat & src, float boundary, float lower, float upper, int boundary2upper)
{
#ifdef MAT_DEBUG
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat mark;
	src.swap(mark);
	switch (boundary2upper)
	{
	case -1:
		for (int ind = 0; ind < mark.length(); ind++)
			mark(ind) = mark(ind) <= boundary ? lower : upper;
		break;
	case 0:
		for (int ind = 0; ind < mark.length(); ind++)
			mark(ind) = mark(ind) >= boundary ? upper : lower;
		break;
	case 1:
		for (int ind = 0; ind < mark.length(); ind++)
			mark(ind) = mark(ind) < boundary ? lower : (mark(ind) == boundary ? boundary : upper);
		break;
	default:
		cerr << errinfo[ERR_INFO_UNLESS] << endl;
		throw std::exception(errinfo[ERR_INFO_UNLESS]);
	}
	return mark;
}
const Mat nn::Filter2D(const Mat & input, const Mat & kern, Point anchor, const Size & strides, bool is_copy_border)
{
	if (input.channels() != 1) {
		fprintf(stderr, "input must be 2D!\n");
		throw Mat();
	}
	if (kern.channels() != 1) {
		fprintf(stderr, "kern must be 2D!\n");
		throw Mat();
	}
	int kern_row = kern.rows();
	int kern_col = kern.cols();
	int left, right, top, bottom;
	Size3 size = mCalSize(input.size3(), kern.size3(), anchor, strides, left, right, top, bottom);
	Mat dst, src;
	if (is_copy_border) {		
		src = copyMakeBorder(input, top, bottom, left, right);
		dst = zeros(input.rows() / strides.h, input.cols() / strides.w);
	}
	else {
		src = input;
		dst = zeros(size.h, size.w);
	}
	for (int row = top, x = 0; row < src.rows() - bottom; row += strides.h, x++)
		for (int col = left, y = 0; col < src.cols() - right; col += strides.w, y++) {
			float value = 0;
			for (int i = 0; i < kern_row; ++i) {
				for (int j = 0; j < kern_col; ++j) {
					value += src(row + i - anchor.x, col + j - anchor.y)*kern(i, j);
				}
			}
			dst(x, y) = value;
		}
	return dst;
}

void nn::Filter2D(const Mat & input, Mat & out, const Mat & kern, Point anchor, const Size & strides, bool is_copy_border)
{
	Mat in;
	int kern_row = kern.rows();
	int kern_col = kern.cols();
	int left, right, top, bottom;
	Size3 size = mCalSize(input.size3(), kern.size3(), anchor, strides, left, right, top, bottom);
	if (is_copy_border) {
		in = copyMakeBorder(input, top, bottom, left, right);
		out = zeros(input.rows() / strides.h, input.cols() / strides.w, size.c);
	}
	else {
		in = input;
		out = zeros(size);
	}
	for (int kern_c = 0; kern_c < size.c; kern_c++) {
		for (int in_c = 0; in_c < input.channels(); in_c++) {
			int c = kern_c * input.channels() + in_c;
			for (int row = top, x = 0; row < in.rows() - bottom; row += strides.h, x++)
				for (int col = left, y = 0; col < in.cols() - right; col += strides.w, y++) {
					int px = col - anchor.y;
					int py = row - anchor.x;
					float value = 0;
					for (int i = 0; i < kern_row; ++i) {
						for (int j = 0; j < kern_col; ++j) {
							value += in(py + i, px + j, in_c)*kern(i, j, c);
						}
					}
					out(x, y, kern_c) += value;
				}
		}
	}
}

const Mat nn::LeastSquare(const Mat & x, const Mat & y)
{
	return (x.t()*x).inv()*x.t()*y;
}

const Mat nn::Reshape(const Mat & src, Size3 size)
{
	Mat dst;
	src.swap(dst);
	dst.reshape(size);
	return dst;
}

const Mat nn::SumChannel(const Mat & src)
{
#ifdef MAT_DEBUG
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	Mat m = zeros(src.mSize());
	for (int i = 0; i < src.rows(); i++)
		for (int j = 0; j < src.cols(); j++) {
			float sum = 0;
			for (int z = 0; z < src.channels(); z++)
				sum += src(i, j, z);
			m(i, j) = sum;
		}
	return m;
}

const Mat nn::rotate(const Mat & src, RotateAngle dice)
{
	Mat dst;
	switch (dice)
	{
	case nn::ROTATE_90_ANGLE:
	{
		Mat mat(src.cols(), src.rows(), src.channels());
		for (int row = 0; row < src.rows(); ++row)
			for (int col = 0; col < src.cols(); ++col)
				for (int depth = 0; depth < src.channels(); ++depth)
					mat(col, src.rows() - 1 - row, depth) = src(row, col, depth);
		dst = mat;
	}
	break;
	case nn::ROTATE_180_ANGLE:
	{
		Mat mat(src.rows(), src.cols(), src.channels());
		for (int row = 0, y = src.rows() - 1; row < src.rows() && y >= 0; ++row, --y)
			for (int col = 0, x = src.cols() - 1; col < src.cols() && x >= 0; ++col, --x)
				for (int depth = 0; depth < src.channels(); ++depth)
					mat(row, col, depth) = src(y, x, depth);
		dst = mat;
	}
	break;
	case nn::ROTATE_270_ANGLE:
	{
		Mat mat(src.cols(), src.rows(), src.channels());
		for (int row = 0; row < src.rows(); ++row)
			for (int col = 0; col < src.cols(); ++col)
				for (int depth = 0; depth < src.channels(); ++depth)
					mat(src.cols() - 1 - col, row, depth) = src(row, col, depth);
		dst = mat;
	}
	break;
	default:
		break;
	}
	return dst;
}

void nn::pause()
{
	fprintf(stderr, "waitting press enter key...\n");
	while (getchar() != '\n');
}

template<typename T>
void nn::showMatrix(const T *temp, int row, int col)
{
	for (int i = 0; i < row; i++) {
		cout << "[ ";
		for (int j = 0; j < col - 1; j++) {
			cout << setw(8) << scientific << setprecision(2) << showpos << left << setfill(' ') << temp(i*col + j) << ", ";
		}
		cout << setw(8) << scientific << setprecision(2) << showpos << left << setfill(' ') << temp(col - 1 + i * col);
		cout << " ]" << endl;
	}
}

