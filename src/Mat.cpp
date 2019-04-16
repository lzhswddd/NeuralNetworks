#include <iostream>
#include <cmath>
#include <time.h>
#include <iomanip>
#include "Mat.h"

using namespace std;
using namespace nn;

const Matrix nn::operator + (const double value, const Matrix &mat)
{
	return mat + value;
}
const Matrix nn::operator - (const double value, const Matrix &mat)
{
	return value + (-mat);
}
const Matrix nn::operator * (const double value, const Matrix &mat)
{
	return mat * value;
}
const Matrix nn::operator / (const double value, const Matrix &mat)
{
	return Divi(mat, value, LEFT);
}
ostream & nn::operator << (ostream &out, const Matrix &ma)
{
	if (ma.matrix == nullptr)
		cout << "error: ¾ØÕóÎª¿Õ" << endl;
	else
		ma.show();
	return out;
}

void nn::check(int row, int col, int depth)
{
	if (row <= 0 || col <= 0 || depth <= 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
}

void nn::Srandom()
{
	srand(uint(time(NULL)));
}

double nn::Max(const Matrix &temp, bool isAbs)
{
	if (isAbs) {
		Matrix m = mAbs(temp);
		return m.findmax();
	}
	else
		return temp.findmax();
}
double nn::Min(const Matrix &temp, bool isAbs)
{
	if (isAbs) {
		Matrix m = mAbs(temp);
		return m.findmin();
	}
	else
		return temp.findmin();
}
double nn::trace(const Matrix &temp)
{
	if (temp.isEnable() == -1){
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if(temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw errinfo[0];
	}
	double sum = 0;
	for (int index = 0; index < temp.rows(); index++) {
		sum += temp((index + index * temp.cols())*temp.channels());
	}
	return sum;
}
double nn::cof(const Matrix &temp, int x, int y)
{
	if (temp.isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (x >= temp.cols() || y >= temp.rows()) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	temp.DimCheck();
	Matrix a(temp.rows() - 1, temp.cols() - 1);
	int n = temp.rows();
	for (int i = 0, k = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if ((i != x) && (j != y)) {
				a(k) = temp(i*n + j);
				k++;
			}
	return det(a);
}
double nn::det(const Matrix &temp)
{
	if (temp.isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	temp.DimCheck();
	if (temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw errinfo[0];
	}
	int n = temp.rows();
	if (n == 1)
		return temp(0);
	else {
		Matrix a(temp);
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++) {
				if (a(j + j * n) == 0) {
					double m;
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
					double w = -a(j + i * n) / a(j + j * n);
					for (int f = j; f < n; f++)
						a(f + i * n) += a(f + j * n) * w;
				}
			}
		double answer = 1;
		for (int i = 0; i < n; i++)
			answer *= a(i + i * n);
		return answer;
	}
}
double nn::getRandData(int min, int max, bool isdouble)
{
	if (min > max) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (isdouble) {
		double m1 = (double)(rand() % 101) / 101;
		min++;
		double m2 = (double)((rand() % (max - min + 1)) + min);
		m2 = m2 - 1;
		return m1 + m2;
	}
	else {
		int m = rand() % (max - min) + 1 + min;
		return (double)m;
	}
}

double nn::mNorm(const Matrix &temp, int num)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (temp.cols() != 1 && temp.rows() != 1) {
		cerr << errinfo[ERR_INFO_NORM] << endl;
		throw errinfo[0];
	}
	temp.DimCheck();
	if (num < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (num == 1)
		return temp.Sum(1, true);
	else if (num == 2)
		return sqrt(temp.Sum(2, true));
	//else if (isinf(num) == 1)
	//	return abs(matrix[find(findmax())]);
	//else if (isinf(num) == -1)
	//	return abs(matrix[find(findmin())]);
	else
		return pow(temp.Sum(num, true), 1 / double(num));
}
double nn::mDistance(const Matrix &a, const Matrix &b, int num)
{
	return (a - b).Norm(num);
}

double nn::mRandSample(const Matrix &m)
{
	int row = rand() % m.rows();
	int col = rand() % m.cols();
	int depth = rand() % m.channels();
	return m(row, col, depth);
}

const Matrix nn::VectoMat(vector<double> &p)
{
	Matrix error;
	if (p.empty())return error;
	Matrix m(int(p.size()), 1);
	for (int iter = 0; iter != int(p.size()); ++iter) {
		m(iter) = p[iter];
	}
	return m;
}
const Matrix nn::VectoMat(vector<vector<double>> &ps)
{
	Matrix error;
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
	Matrix m(hei, wid);
	for (int i = 0; i < hei; ++i) {
		for (int j = 0; j < wid; ++j) {
			m(i, j) = ps[i][j];
		}
	}
	return m;
}
vector<double> nn::MattoVec(const Matrix & m)
{
	if (m.empty())return vector<double>();
	vector<double> p(m.size());
	for (int iter = 0; iter != m.size(); ++iter) {
		p[iter] = m(iter);
	}
	return p;
}

vector<vector<double>> nn::MattoVecs(const Matrix & m)
{
	if (m.empty())return vector<vector<double>>();
	vector<vector<double>> ps;
	for (int row = 0; row != m.rows(); ++row) {
		vector<double> p;
		for (int col = 0; col != m.cols(); ++col) {
			p.push_back(m(row, col));
		}
		ps.push_back(p);
	}
	return ps;  
}

const Matrix nn::eye(int n)
{
	check(n, n);
	Matrix mark(n, n);
	for (int ind = 0; ind < n; ind++)
		mark(ind + ind * n) = 1;
	return mark;
}
const Matrix nn::mSplit(const Matrix & src, int channel)
{
	check(src.rows(), src.cols(), src.channels());
	if (channel > src.channels() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (channel < 0) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Mat mat(src.rows(), src.cols());
	for (int i = 0; i <src.rows(); i++)
		for (int j = 0; j < src.cols(); j++) {
			mat(i, j) = src(i, j, channel);
		}
	return mat;
}
void nn::mSplit(const Matrix & src, Matrix * dst)
{
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	for (int channel = 0; channel < src.channels(); ++channel)
		dst[channel] = src[channel];
}
const Matrix nn::mMerge(const Matrix * src, int channels)
{
	if (channels < 0) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (src == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (src[channels - 1].empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mat(src[0].rows(), src[0].cols(), channels);
	for (int z = 0; z < channels; z++) {
		for (int i = 0; i <src[z].rows(); i++)
			for (int j = 0; j < src[z].cols(); j++) {
				mat(i, j, z) = src[z](i, j);
			}
	}
	return mat;
}
const Matrix nn::zeros(int row, int col)
{
	check(row, col);
	Matrix mat(row, col);
	return mat;
}
const Matrix nn::zeros(int row, int col, int channel)
{
	check(row, col, channel);
	Matrix mat(row, col, channel);
	return mat;
}
const Matrix nn::zeros(Size size)
{
	check(size.hei, size.wid);
	Matrix mat(size.hei, size.wid);
	return mat;
}
const Matrix nn::zeros(Size3 size)
{
	return Matrix(size);
}
const Matrix nn::value(double v, int row, int col, int channel)
{
	check(row, col, channel);
	Matrix mark(row, col, channel);
	for (int ind = 0; ind < row*col*channel; ind++)
		mark(ind) = v;
	return mark;
}
const Matrix nn::ones(int row, int col)
{
	return value(1, row, col);
}
const Matrix nn::ones(int row, int col, int channel)
{
	return value(1, row, col, channel);
}
const Matrix nn::ones(Size size)
{
	check(size.hei, size.wid);
	return value(1, size.hei, size.wid);
}
const Matrix nn::reverse(const Matrix &m)
{
	if (!(m.cols() == 1 || m.rows() == 1)) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (m.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix temp(m);
	for (int ind = 0; ind < m.size() / 2; ind++) {
		double val = temp(ind);
		temp(ind) = temp(m.size() - 1 - ind);
		temp(m.size() - 1 - ind) = val;
	}
	return temp;
}
const Matrix nn::mRandSample(const Matrix &src, int row, int col, int channel)
{
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	check(row, col, channel);
	Matrix dst(row, col, channel);
	for (int ind = 0; ind < src.size(); ind++)
		dst(ind) = mRandSample(src);
	return dst;
}
const Matrix nn::mRandSample(const Matrix& m, X_Y_Z rc, int num)
{
	if (m.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix dst = m(rand() % m.rows(), rc);
	for (int i = 1; i < num; i++) {
		dst = Matrix(dst, m(rand() % m.rows(), rc), rc);
	}
	return dst;
}
const Matrix nn::linspace(int low, int top, int len)
{
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	check(len, len);
	Matrix mark(1, len);
	mark(0)= low;
	double gap = double(abs(low) + abs(top)) / (len - 1);;
	for (int ind = 1; ind < len; ind++)
		mark(ind) = mark(ind - 1) + gap;
	return mark;
}
const Matrix nn::linspace(double low, double top, int len)
{
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	check(len, len);
	Matrix mark(1, len);
	double gap = (top - low) / (len - 1);
	mark = low + linspace(0, len - 1, len)*gap;
	if (mark.isEnable() != -1) {
		mark(0) = low;
		mark(len - 1) = top;
	}
	return mark;
}
const Matrix nn::copyMakeBorder(const Matrix & src, int top, int bottom, int left, int right, BorderTypes borderType, const int value)
{
	Size3 size = src.size3();
	size.x += (top + bottom);
	size.y += (left + right);
	Matrix mat(size);
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
		for (int i = top; i < src.rows(); i++) {
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
		for (int i = top; i < src.rows(); i++) {
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
const Matrix nn::Block(const Matrix&a, int Row_Start, int Row_End, int Col_Start, int Col_End)
{
	int hei = Row_End - Row_Start + 1;
	int wid = Col_End - Col_Start + 1;
	check(hei, wid);
	Matrix mark(hei, wid);
	int i = 0;
	for (int y = Row_Start, j = 0; y <= Row_End; y++, j++)
		for (int x = Col_Start, i = 0; x <= Col_End; x++, i++)
			mark(i + j * wid) = a(y, x);
	return mark;
}
const Matrix nn::mRand(int low, int top, int n, bool isdouble)
{
	return mRand(low, top, n, n, 1, isdouble);
}
const Matrix nn::mRand(int low, int top, int row, int col, int channel, bool isdouble)
{
	check(row, col, channel);
	if (low >= top) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	Matrix m(row, col, channel);
	for (int index = 0; index < m.length(); index++)
		m(index) = getRandData(low, top, isdouble);
	return m;
}
const Matrix nn::mcreate(int row, int col)
{
	return Matrix(row, col);
}
const Matrix nn::mcreate(int row, int col, int channel)
{
	return Matrix(row, col, channel);
}
const Matrix nn::mcreate(Size size)
{
	return Matrix(size);
}
const Matrix nn::mcreate(Size3 size)
{
	return Matrix(size);
}
const Matrix nn::adj(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	if (temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw temp;
	}
	int n = temp.rows();
	int depth = temp.channels();
	Matrix a(n, n, depth);
	for (int z = 0; z < depth; z++) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++) {
				double m = cof(temp, i, j);
				a((i*n + j)*depth + z) = pow(-1, i + j + 2)*m;
			}
	}
	return tran(a);
}
const Matrix nn::inv(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix *m = new Matrix[temp.channels()];
	for (int z = 0; z < temp.channels(); z++) {
		m[z] = temp[z];
		double answer = det(m[z]);
		if (answer != 0 && answer == answer) {
			m[z] = adj(m[z]);
			int n = m[z].rows();
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++)
					m[z](i, j) = (1 / answer)*m[z](i, j);
		}
		else {
			cerr << errinfo[ERR_INFO_DET] << endl;
			throw temp;
		}
	}
	Matrix mat = mMerge(m, temp.channels());
	delete[] m;
	return mat;
}
const Matrix nn::pinv(const Matrix &temp, direction direc)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	switch (direc)
	{
	case LEFT:return (temp.t()*temp).Inv()*temp.t();
	case RIGHT: {
		Matrix m = temp.t();
		return nn::pinv(m, LEFT).t();
	}
	default: 
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw direc;
	}
}
const Matrix nn::tran(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix a(temp.cols(), temp.rows(), temp.channels());
	int n = temp.rows(),
		m = temp.cols();
	for (int z = 0; z < temp.channels(); z++)
		for (int i = 0; i < n; i++)
			for (int j = 0; j < m; j++)
				a(j, i, z) = temp(i, j, z);
	return a;
}
const Matrix nn::mAbs(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix m(temp);
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = fabs(temp(ind));
	return m;
}
const Matrix nn::Rotate(double angle)
{
	Matrix rotate_mat(2, 2);
	angle = angle * pi / 180.0f;
	rotate_mat(0) = cos(angle);
	rotate_mat(1) = -sin(angle);
	rotate_mat(2) = sin(angle);
	rotate_mat(3)= cos(angle);
	return rotate_mat;
}
const Matrix nn::POW(const Matrix &temp, int num)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	if (temp.isEnable() == -2) {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw temp;
	}
	else {
		Matrix m(temp);
		if (num > 0) {
			for (int i = 1; i < num; i++)
				m = m * temp;
			return m;
		}
		else if (num < 0) {
			Matrix a(temp);
			m.setInv();
			a.setInv();
			for (int i = -1; i > num; i--)
				a = a * m;
			return a;
		}
		else
			return eye(temp.rows());
	}
}
const Matrix nn::mPow(const Matrix &temp, int num)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix m(temp);
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = pow(temp(ind), num);
	return m;
}
const Matrix nn::mSum(const Matrix &temp, X_Y_Z r_c)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	if (r_c == COL) {
		Matrix m(temp.channels(), temp.cols());
		for (int z = 0; z < temp.channels(); z++)
			for (int i = 0; i < temp.cols(); i++)
				for (int j = 0; j < temp.rows(); j++)
					m(z, i) += temp(j, i, z);
		return m;
	}
	else if (r_c == ROW) {
		Matrix m(temp.rows(), temp.channels());
		for (int z = 0; z < temp.channels(); z++)
			for (int i = 0; i < temp.rows(); i++)
				for (int j = 0; j < temp.cols(); j++)
					m(i, z) += temp(i, j, z);
		return m;
	}
	else if (r_c == CHANNEL) {
		Matrix m(1, 1, temp.channels());
		for (int z = 0; z < temp.channels(); z++) {
			double sum = 0;
			for (int i = 0; i < temp.rows(); i++)
				for (int j = 0; j < temp.cols(); j++)
					sum += temp(i, j, z);
			m(z) = sum;
		}
		return m;
	}
	else {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw r_c;
	}
}
const Matrix nn::mExp(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++) {
		m(ind) = exp(temp(ind));
		if (m(ind) == 0)
			m(ind) = (numeric_limits<double>::min)();
	}
	return m;
}
const Matrix nn::mLog(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix m(temp.size3());
	for (int ind = 0; ind < temp.length(); ind++) 
		if (temp(ind) == 0)
			m(ind) = (numeric_limits<double>::min)();
		else
			m(ind) = log(temp(ind));
	return m;
}
const Matrix nn::mSqrt(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix m(temp);
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = sqrt(temp(ind));
	return m;
}
const Matrix nn::mOpp(const Matrix &temp)
{
	if (temp.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw temp;
	}
	Matrix m(temp);
	for (int ind = 0; ind < temp.length(); ind++)
		m(ind) = -temp(ind);
	return m;
}
const Matrix nn::Divi(const Matrix &a, double val, direction dire)
{
	if (a.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(a.rows(), a.cols(), a.channels());
	for (int ind = 0; ind < mark.length(); ind++)
			if (dire == LEFT)
				mark(ind) = val / a(ind);
			else if (dire == RIGHT)
				mark(ind) = a(ind) / val;
	return mark;
}
const Matrix nn::Divi(const Matrix &a, const Matrix &b, direction dire)
{
	switch (dire)
	{
	case LEFT:return a.Inv()*b;
	case RIGHT:return a / b;
	default:
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw dire;
	}
}
const Matrix nn::Mult(const Matrix &a, const Matrix &b)
{
	if (a.isEnable() == -1 || b.isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (a.rows() != b.rows() || a.cols() != b.cols() || a.channels() != b.channels()) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw errinfo[0];
	}
	Matrix temp(a.rows(), a.cols(), a.channels());
	for (int ind = 0; ind < a.length(); ind++)
		temp(ind) = a(ind) * b(ind);
	return temp;
}
const Matrix nn::mMax(double a, const Matrix &b)
{
	if (b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a > b(ind) ? a : b(ind);
	return mark;
}
const Matrix nn::mMax(const Matrix &a, const Matrix &b)
{
	if (a.empty()|| b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (a.rows() != b.rows() || a.cols() != b.cols() || a.channels() != b.channels()) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw errinfo[0];
	}
	Matrix mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a(ind) > b(ind) ? a(ind) : b(ind);
	return mark;
}
const Matrix nn::mMin(double a, const Matrix &b)
{
	if (b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a < b(ind) ? a : b(ind);
	return mark;
}
const Matrix nn::mMin(const Matrix &a, const Matrix &b)
{
	if (a.empty() || b.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (a.rows() != b.rows() || a.cols() != b.cols() || a.channels() != b.channels()) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw errinfo[0];
	}
	Matrix mark(b.rows(), b.cols(), b.channels());
	for (int ind = 0; ind < b.length(); ind++)
		mark(ind) = a(ind) < b(ind) ? a(ind) : b(ind);
	return mark;
}
Size3 nn::mCalSize(const Matrix & src, const Matrix & kern, Point & anchor, Size strides, int & top, int & bottom, int & left, int & right)
{
	int kern_row = kern.rows();
	int kern_col = kern.cols();
	if (anchor == Point(-1, -1)) {
		anchor.x = kern_row % 2 ? kern_row / 2 : kern_row / 2 - 1;
		anchor.y = kern_col % 2 ? kern_col / 2 : kern_col / 2 - 1;
	}
	top = anchor.x;
	bottom = kern_row - anchor.x - 1;
	left = anchor.y;
	right = kern_col - anchor.y - 1;
	return Size3((src.rows() - top - bottom) / strides.hei, (src.cols() - left - right) / strides.wid, kern.channels()/ src.channels());
}
Size3 nn::mCalSize(Size3 src, Size3 kern, Point & anchor, Size strides)
{
	int kern_row = kern.y;
	int kern_col = kern.x;
	if (anchor == Point(-1, -1)) {
		anchor.x = kern_row % 2 ? kern_row / 2 : kern_row / 2 - 1;
		anchor.y = kern_col % 2 ? kern_col / 2 : kern_col / 2 - 1;
	}
	int top = anchor.x;
	int bottom = kern_row - anchor.x - 1;
	int left = anchor.y;
	int right = kern_col - anchor.y - 1;
	return Size3((src.y - top - bottom) / strides.hei, (src.x - left - right) / strides.wid, kern.z / src.z);
}
const Matrix nn::mThreshold(const Matrix & src, double boundary, double lower, double upper, int boundary2upper)
{
	if (src.empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(src);
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
		throw errinfo[0];
	}
	return mark;
}
const Matrix nn::Filter2D(const Mat & input, const Mat & kern, Point anchor, const Size & strides, bool is_copy_border)
{
	if (input.channels() != 1) {
		fprintf(stderr, "input must be 2D!");
		throw Mat();
	}
	if (kern.channels() != 1) {
		fprintf(stderr, "kern must be 2D!");
		throw Mat();
	}
	Mat src;
	int kern_row = kern.rows();
	int kern_col = kern.cols();
	int left, right, top, bottom;
	Size3 size = mCalSize(input, kern, anchor, strides, left, right, top, bottom);
	Mat dst;
	if (is_copy_border) {		
		src = copyMakeBorder(input, top, bottom, left, right);
		dst = zeros(input.rows() / strides.hei, input.cols() / strides.wid);
	}
	else {
		input.swap(src);
		dst = zeros(size.x, size.y);
	}
	for (int row = top, x = 0; row < src.rows() - bottom; row += (int)strides.hei, x++)
		for (int col = left, y = 0; col < src.cols() - right; col += (int)strides.wid, y++) {
			double value = 0;
			for (int i = 0; i < kern_row; ++i) {
				for (int j = 0; j < kern_col; ++j) {
					value += src(row + i - anchor.x, col + j - anchor.y)*kern(i, j);
				}
			}
			dst(x, y) = value;
		}
	return dst;
}

const Matrix nn::LeastSquare(const Matrix & x, const Matrix & y)
{
	return (x.t()*x).Inv()*x.t()*y;
}

void nn::pause()
{
	fprintf(stderr, "waitting press enter key...");
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

