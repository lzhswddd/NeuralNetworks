#include <iostream>
#include <fstream>
#include <iomanip>
#include "alignMalloc.h"
#include "Function.h"
#include "Mat.h"
#include "Matrix.h"
using namespace std;
using namespace nn;


Matrix::Matrix()
{
	init();
	checkSquare();
}
Matrix::Matrix(int row, int col)
{
	init();
	check(row, col);
	channel = 1;
	this->row = row;
	this->col = col;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	checkSquare();
}
Matrix::Matrix(int row, int col, int depth)
{
	init();
	check(row, col);
	this->row = row;
	this->col = col;
	this->channel = depth;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*depth * sizeof(float));
	//memset(matrix, 0, row*col*channel * sizeof(float));
	checkSquare();
}
Matrix::Matrix(Size size_)
{
	init();
	check(size_.hei, size_.wid);
	row = size_.hei;
	col = size_.wid;
	channel = 1;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	//memset(matrix, 0, row*col*channel * sizeof(float));
	checkSquare();
}
Matrix::Matrix(Size3 size_)
{
	init();
	check(size_.x, size_.y, size_.z);
	row = size_.x;
	col = size_.y;
	channel = size_.z;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	//memset(matrix, 0, row*col*channel * sizeof(float));
	checkSquare();
}
Matrix::Matrix(float *matrix, int n)
{
	init();
	*this = Matrix(matrix, n, n);
}
Matrix::Matrix(int *matrix, int n)
{
	init();
	*this = Matrix(matrix, n, n);
}
Matrix::Matrix(int *matrix, int row, int col, int channel)
{
	init();
	if (matrix != nullptr) {
		check(row, col, channel);
		channel = 1;
		this->row = row;
		this->col = col;
		this->channel = channel;
#ifdef LIGHT_MAT
		createCount();
#endif
		this->matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
		if (this->matrix != nullptr)
			for (int index = 0; index < length(); index++)
				this->matrix[index] = (float)matrix[index];
	}
	checkSquare();
}
Matrix::Matrix(float *matrix, int row, int col, int channel)
{
	init();
	if (matrix != nullptr) {
		check(row, col);
		channel = 1;
		this->row = row;
		this->col = col;
		this->channel = channel;
#ifdef LIGHT_MAT
		createCount();
#endif
		this->matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
		if (this->matrix != nullptr)
			memcpy(this->matrix, matrix, row*col*channel * sizeof(float));
	}
	checkSquare();
}
Matrix::Matrix(int w, float * data)
{
	init();
	col = w;
	row = 1;
	channel = 1;
	matrix = data;
}
Matrix::Matrix(int w, int h, float * data)
{
	init();
	col = w;
	row = h;
	channel = 1;
	matrix = data;
}
Matrix::Matrix(int w, int h, int c, float * data)
{
	init();
	col = w;
	row = h;
	channel = c;
	matrix = data;
}
Matrix::Matrix(const Matrix &src)
{
	init();
	*this = src;
	/*init();
	setvalue(src);
	checkSquare();*/
}
Matrix::Matrix(const Matrix *src)
{
	init();
	*this = *src;
	/*init();
	if (src != nullptr)
		setvalue(*src);
	checkSquare();*/
}
Matrix::Matrix(Matrix a, Matrix b, X_Y_Z merge)
{
	init();
	if (merge == ROW) {
		if (a.col == b.col) {
			row = a.row + b.row;
			col = a.col;
			channel = a.channel;
#ifdef LIGHT_MAT
			createCount();
#endif
			matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
			if (matrix != nullptr) {
				memcpy(matrix, a.matrix,
					a.row*a.col*a.channel * sizeof(float));
				memcpy(matrix + (a.row*a.col*a.channel), b.matrix,
					b.row*b.col*b.channel * sizeof(float));
			}
			else {
				matrix = nullptr;
			}
		}
	}
	else if (merge == COL) {
		if (a.row == b.row) {
			row = a.row;
			col = a.col + b.col;
			channel = a.channel;
#ifdef LIGHT_MAT
			createCount();
#endif
			matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
			float *temp = a.matrix;
			if (matrix != nullptr)
				for (int i = 0; i < row; i++) {
					memcpy(matrix + i * col*channel,
						a.matrix + i * a.col*channel,
						a.col*channel * sizeof(float));
					memcpy(matrix + i * col*channel + a.col*channel,
						b.matrix + i * b.col*channel,
						b.col*channel * sizeof(float));
				}
			else {
				matrix = nullptr;
			}
		}
	}
	checkSquare();
}
Matrix::Matrix(MatCommaInitializer_ & m)
{
	init();
	*this = Matrix(m.matrix(), m.rows(), m.cols(), m.channels());
}
Matrix::~Matrix()
{
	release();
	col = 0;
	row = 0;
	channel = 0;
	/*if (matrix != nullptr) {
		fastFree(matrix);
		matrix = nullptr;
	}*/
}

void Matrix::create(int w)
{
	release();
	check(w,1, 1);
	row = 1;
	col = w;
	channel = 1;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	checkSquare();
}

void Matrix::create(int h, int w)
{
	release();
	check(h, w, 1);
	row = h;
	col = w;
	channel = 1;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	checkSquare();
}

void Matrix::create(int h, int w, int c)
{
	release();
	check(h, w, c);
	row = h;
	col = w;
	channel = c;
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	checkSquare();
}

float* Matrix::mat_()const
{
	return matrix;
}

void Matrix::DimCheck()const
{
	if (channel != 1) {
		cerr << errinfo[ERR_INFO_DIM] << endl;
		throw errinfo[0];
	}
}

void Matrix::copy(Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int hei = Row_End - Row_Start + 1;
	int wid = Col_End - Col_Start + 1;
	check(hei, wid);
	if (src.matrix == nullptr) {
		src = zeros(hei, wid, channel);
	}
	for (int y = Row_Start, j = 0; y <= Row_End; y++, j++)
		for (int x = Col_Start, i = 0; x <= Col_End; x++, i++)
			for (int z = 0; z < channel; z++)
				src(y, x, z) = matrix[(i + j * wid)*channel + z];
}

void Matrix::swap(Matrix &src)const
{
	src.setvalue(*this);
}

void Matrix::addones(direction dire)
{
	Matrix temp(row, col + 1, channel);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col + 1; j++) {
			for (int z = 0; z < channel; z++) {
				if (dire == LEFT) {
					if (j == 0)
						temp(i, j, z) = 1;
					else
						temp(i, j, z) = matrix[(j - 1 + i * col)*channel + z];
				}
				else if (dire == RIGHT) {
					if (j == col)
						temp(i, j, z) = 1;
					else
						temp(i, j, z) = matrix[(j - 1 + i * col)*channel + z];
				}
			}
		}
	}
	*this = temp;
}
void Matrix::mChannel(const Matrix & src, int channels)
{
	if (matrix == nullptr || src.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row != src.row || col != src.col || channels >= channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			matrix[(i*col + j)*channel + channels] = src(i, j);
		}
	}
}
void nn::Matrix::mChannel(const Matrix & src, int row, int col)
{
	if (matrix == nullptr || src.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (this->row <= row || this->col <= col || src.channels() != channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	for (int i = 0; i < channel; ++i) {
		matrix[(row*col + col)*channel + i] = src(i);
	}
}
void nn::Matrix::reshape(Size3 size)
{
	if (matrix == nullptr) {
		fprintf(stderr, errinfo[ERR_INFO_EMPTY]);
		throw errinfo[ERR_INFO_EMPTY];
	}
	if (length() != size.x * size.y*size.z) {
		fprintf(stderr, errinfo[ERR_INFO_UNLESS]);
		throw errinfo[ERR_INFO_UNLESS];
	}
	else {
		this->row = size.x;
		this->col = size.y;
		this->channel = size.z;
	}
}
void Matrix::reshape(int row, int col, int channels)
{
	if (matrix == nullptr) {
		fprintf(stderr, errinfo[ERR_INFO_EMPTY]);
		throw errinfo[ERR_INFO_EMPTY];
	}
	if (length() != row * col*channels) {
		fprintf(stderr, errinfo[ERR_INFO_UNLESS]);
		throw errinfo[ERR_INFO_UNLESS];
	}
	else {
		this->row = row;
		this->col = col;
		this->channel = channels;
	}
}
bool Matrix::setSize(int row, int col, int channel)
{
	if (row*col*channel > 0) {
		*this = Matrix(row, col, channel);
		return true;
	}
	if (length() == row * col * channel) {
		this->row = row;
		this->col = col;
		this->channel = channel;
		return true;
	}
	return false;
}
void Matrix::setNum(float number, int index)
{
	checkindex(index);
	matrix[index] = number;
}
void Matrix::setNum(float number, int index_y, int index_x)
{
	checkindex(index_x, index_y);
	matrix[index_y*col + index_x] = number;
}
void Matrix::setMat(float *mat, int hei, int wid)
{
	if ((row <= 0 || col <= 0)) return;
	*this = Matrix(mat, hei, wid);
	checkSquare();
}
void Matrix::setvalue(const Matrix &src)
{
	row = src.row;
	col = src.col;
	channel = src.channel;
	square = src.square;
	if (src.matrix != nullptr) {
#ifdef LIGHT_MAT
		if (recount != nullptr) {
			if (*recount != 0) {
				*recount -= 1;
			}
			else {
#endif // LIGHT_MAT
				if (matrix != nullptr) {
					fastFree(matrix);
					matrix = nullptr;
				}

#ifdef LIGHT_MAT
			}
		}
		else
			recount = new int(0);
#endif // LIGHT_MAT
		matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
		if (matrix != nullptr)
			memcpy(matrix, src.matrix, row*col*channel * sizeof(float));
	}
	else matrix = nullptr;
}
void Matrix::setOpp()
{
	*this = mOpp(*this);
}
void Matrix::setAdj()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = Adj();
	else {
		cerr << errinfo[ERR_INFO_ADJ] << endl;
		throw errinfo[0];
	}
}
void Matrix::setTran()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	*this = t();
}
void Matrix::setInv()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = Inv();
	else {
		cerr << errinfo[ERR_INFO_INV] << endl;
		throw errinfo[0];
	}
}
void Matrix::setPow(int num)
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = Pow(num);
	else {
		cerr << errinfo[ERR_INFO_POW] << endl;
		throw errinfo[0];
	}
}
void Matrix::setIden()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = eye(row);
	else {
		cerr << errinfo[ERR_INFO_IND] << endl;
		throw errinfo[0];
	}
}
Size3 Matrix::size3() const
{
	return Size3(row, col, channel);
}
int Matrix::rows()const
{
	return row;
}
int Matrix::cols()const
{
	return col;
}
int Matrix::channels() const
{
	return channel;
}
size_t Matrix::size()const
{
	return (size_t)length();
}
Size Matrix::mSize()const
{
	return Size(row, col);
}
void Matrix::save(std::string file, bool binary) const
{
	if (binary) {
		FILE *out = fopen(file.c_str(), "wb");
		if (out) {
			int param[3] = { row, col, channel };
			fwrite(param, sizeof(int) * 3, 1, out);
			fwrite(matrix, sizeof(float)*length(), 1, out);
			fclose(out);
		}
	}
	else {
		ofstream out(file);
		if (out.is_open()) {
			out.setf(ios::scientific);
			out.setf(ios::showpos);
			out.setf(ios::left);
			for (int i = 0; i < row; i++) {
				out << "[ ";
				for (int j = 0; j < col; j++) {
					out << "[ ";
					for (int k = 0; k < channel; k++) {
						if (j == col - 1 && k == channel - 1)
						{
							out << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + j)*channel + k] << " ]]";
						}
						else if (k == channel - 1)
						{
							out << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + j)*channel + k] << " ]";
						}
						else {
							out << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + j)*channel + k] << ", ";
						}
					}
				}
				out << endl;
			}
			out.close();
		}
	}
}
void Matrix::load(std::string file)
{
	FILE *in = fopen(file.c_str(), "rb");
	if (in) {
		int param[3] = { 0 };
		fread(param, sizeof(int) * 3, 1, in);
		create(param[0], param[1], param[2]);
		fread(matrix, sizeof(float)*length(), 1, in);
		fclose(in);
	}
}
int Matrix::length()const
{
	return row * col * channel;
}
int Matrix::isEnable()const
{
	if (matrix == nullptr)
		return -1;
	if (!square)
		return -2;
	return 0;
}
bool Matrix::empty()const
{
	if (matrix == nullptr)return true;
	else return false;
}
bool Matrix::Square()const
{
	return square;
}
void Matrix::release()
{
#ifdef LIGHT_MAT
	if (recount != 0) {
		if (*recount == 0) {
			if (matrix != nullptr) {
				fastFree(matrix);
				matrix = nullptr;
			}
			delete recount;
			recount = nullptr;
		}
		else {
			*recount -= 1;
		}
	}
#else
	if (matrix != nullptr) {
		fastFree(matrix);
		matrix = nullptr;
	}
#endif
}
float& Matrix::at(int index_y, int index_x)const
{
	checkindex(index_x, index_y);
	return matrix[index_y*col + index_x];
}
float& Matrix::at(int index)const
{
	checkindex(index);
	return matrix[index];
}
int Matrix::toX(int index)const
{
	return index % col;
}
int Matrix::toY(int index)const
{
	return index / col;
}
float Matrix::frist()const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[0];
}
float& Matrix::findAt(float value)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	static float err = NAN;
	for (int ind = 0; ind < length(); ind++)
		if (matrix[ind] == value)
			return matrix[ind];
	return err;
}
float& Matrix::findmax()const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int max_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[max_adr] < matrix[ind])
			max_adr = ind;
	return matrix[max_adr];
}
float& Matrix::findmin()const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int min_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[min_adr] > matrix[ind])
			min_adr = ind;
	return matrix[min_adr];
}
int Matrix::find(float value)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	for (int ind = 0; ind < length(); ind++)
		if (matrix[ind] == value)
			return ind;
	return -1;
}
int Matrix::maxAt()const
{
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int max_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[max_adr] < matrix[ind])
			max_adr = ind;
	return max_adr;
}
int Matrix::minAt()const
{
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int min_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[min_adr] > matrix[ind])
			min_adr = ind;
	return min_adr;
}
bool Matrix::contains(float value)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	for (int ind = 0; ind < length(); ind++)
		if (matrix[ind] == value)
			return true;
	return false;
}

void Matrix::show()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	cout.setf(ios::scientific);
	cout.setf(ios::showpos);
	cout.setf(ios::left);
	for (int z = 0; z < channel; z++) {
		for (int i = 0; i < row; i++) {
			cout << "[ ";
			for (int j = 0; j < col - 1; j++) {
				cout << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + j)*channel + z] << ", ";
			}
			cout << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + col - 1)*channel + z];
			cout << " ]";
		}
	}
	cout.unsetf(ios::scientific);
	cout.unsetf(ios::showpos);
	cout.unsetf(ios::left);
	cout << defaultfloat << setprecision(6);
}

const Matrix Matrix::Abs()const
{
	return mAbs(*this);
}
const Matrix nn::Matrix::mPow(int num) const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		return POW(*this, num);
	}
	else{
		cerr << errinfo[ERR_INFO_POW] << endl;
		throw errinfo[0];
	}
}
const Matrix Matrix::Pow(int num)const
{
	return nn::mPow(*this, num);
}
float Matrix::Sum(int num, bool _abs)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	float sum = 0;
	if (num == 1) {
		for (int ind = 0; ind < length(); ind++)
			if (_abs)
				sum += fabs(matrix[ind]);
			else
				sum += matrix[ind];
	}
	else
		for (int ind = 0; ind < length(); ind++)
			if (_abs)
				sum += pow(fabs(matrix[ind]), num);
			else
				sum += pow(matrix[ind], num);
	return sum;
}
const Matrix nn::Matrix::clone() const
{
	Matrix dst;
	dst.setvalue(*this);
	return dst;
}
const Matrix Matrix::Opp()const
{
	return mOpp(*this);
}
const Matrix Matrix::Adj()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		return adj(*this);
	}
	else {
		cerr << errinfo[ERR_INFO_ADJ] << endl;
		throw errinfo[0];
	}
}
const Matrix Matrix::t()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return tran(*this);
}
const Matrix Matrix::Inv()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		return inv(*this);
	}
	else {
		try {
			return pinv(*this, RIGHT);
		}
		catch (...) {
			try {
				return pinv(*this, LEFT);
			}
			catch (...) {
				cerr << errinfo[ERR_INFO_PINV] << endl;
				throw errinfo[0];
			}
		}
	}
}
const Matrix Matrix::Reverse()const
{
	return reverse(*this);
}
float Matrix::Det()
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square)
		return det(*this);
	else
		return NAN;
}
float Matrix::Norm(int num)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (col != 1 && row != 1) {
		cerr << errinfo[ERR_INFO_NORM] << endl;
		throw errinfo[0];
	}
	if (num < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (num == 1)
		return Sum(1, true);
	else if (num == 2)
		return sqrt(Sum(2, true));
	//else if (isinf(num) == 1)
	//	return abs(matrix[find(findmax())]);
	//else if (isinf(num) == -1)
	//	return abs(matrix[find(findmin())]);
	else
		return pow(Sum(num, true), 1 / float(num));
}
float Matrix::Matrix::Cof(int x, int y)
{
	return cof(*this, x, y);
}
float Matrix::EigenvalueMax(float offset)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		int count = 0;
		float err = 100 * offset;
		Matrix v;
		Matrix u0 = ones(row, 1);
		while (err > offset) {
			v = *this*u0;
			Matrix u1 = v * (1 / v.findmax());
			err = (u1 - u0).Abs().findmax();
			u0 = u1; count += 1;
			if (count >= 1e+3) {
				cerr << errinfo[ERR_INFO_EIGEN] << endl;
				throw errinfo[0];
			}
		}
		return v.findmax();
	}
	else {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw errinfo[0];
	}
}
float Matrix::RandSample()
{
	return mRandSample(*this);
}
const Matrix Matrix::EigenvectorsMax(float offset)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		int count = 0;
		float err = 100 * offset;
		Matrix v;
		Matrix u0 = ones(row, 1);
		while (err > offset) {
			v = *this*u0;
			Matrix u1 = v * (1 / v.findmax());
			err = (u1 - u0).Abs().findmax();
			u0 = u1; count += 1;
			if (count >= 1e+3) {
				cerr << errinfo[ERR_INFO_EIGEN] << endl;
				throw errinfo[0];
			}
		}
		return u0;
	}
	else {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw errinfo[0];
	}
}

const Matrix nn::Matrix::sigmoid() const
{
	return Sigmoid(*this);
}

const Matrix nn::Matrix::tanh() const
{
	return Tanh(*this);
}

const Matrix nn::Matrix::relu() const
{
	return ReLU(*this);
}

const Matrix nn::Matrix::elu() const
{
	return ELU(*this);
}

const Matrix nn::Matrix::selu() const
{
	return SELU(*this);
}

const Matrix nn::Matrix::leaky_relu() const
{
	return LReLU(*this);
}

const Matrix nn::Matrix::softmax() const
{
	return Softmax(*this);
}

const Matrix Matrix::Exp()const
{
	return mExp(*this);
}

const Matrix Matrix::Log()const
{
	return mLog(*this);
}

const Matrix Matrix::Sqrt()const
{
	return mSqrt(*this);
}

void Matrix::init()
{
	row = 0;
	col = 0;
	channel = 0;
	matrix = nullptr;
#ifdef LIGHT_MAT
	recount = 0;
#endif // LIGHT_MAT
}

#ifdef LIGHT_MAT
void nn::Matrix::createCount()
{
	recount = new int(0);
}
#endif
void Matrix::checkSquare()
{
	if (row == col)
		square = true;
	else
		square = false;
}
void Matrix::checkindex(int index)const
{
	if (row == 0 || col == 0) {
		cerr << errinfo[ERR_INFO_LEN] << endl;
		throw errinfo[0];
	}
	if (index > length() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
}
void Matrix::checkindex(int index_x, int index_y)const
{
	if (row == 0 || col == 0) {
		cerr << errinfo[ERR_INFO_LEN] << endl;
		throw errinfo[0];
	}
	if (index_x < 0 || index_y < 0) {
		cerr << errinfo[ERR_INFO_UNLESS] << endl;
		throw errinfo[0];
	}
	if (index_x*col + index_y > row*col - 1 || index_x >= row || index_y >= col) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
}

const Matrix Matrix::operator + (const float val)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] + val;
	return mark;
}
const Matrix Matrix::operator + (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && channel == 1) {
		return (*this)(0) + a;
	}
	else if (a.row == 1 && a.col == 1 && a.channel == 1) {
		return *this + a(0);
	}
	else if (a.row == 1 && a.col == 1 && a.channel == channel) {
		Matrix mat(row, col, channel);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				for (int z = 0; z < channel; z++)
					mat((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] + a(z);
		return mat;
	}
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] + a((j + i * col)*channel + z);
	return mark;
}
void Matrix::operator += (const float val)
{
	*this = *this + val;
}
void Matrix::operator += (const Matrix & a)
{
	*this = *this + a;
}
const Matrix Matrix::operator-(void) const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = -matrix[(j + i * col)*channel + z];
	return mark;
}
const Matrix Matrix::operator - (const float val)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] - val;
	return mark;
}
const Matrix Matrix::operator - (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && channel == 1) {
		return (*this)(0) - a;
	}
	else if (a.row == 1 && a.col == 1 && a.channel == 1) {
		return *this - a(0);
	}
	else if (a.row == 1 && a.col == 1 && a.channel == channel) {
		Matrix mat(row, col, channel);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				for (int z = 0; z < channel; z++)
					mat((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] - a(z);
		return mat;
	}
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] - a((j + i * col)*channel + z);
	return mark;
}
void Matrix::operator-=(const float val)
{
	*this = *this - val;
}
void Matrix::operator-=(const Matrix & a)
{
	*this = *this - a;
}
const Matrix Matrix::operator * (const float val)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] * val;
	return mark;
}
const Matrix Matrix::operator * (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && channel == 1) {
		return (*this)(0) * a;
	}
	else if (a.row == 1 && a.col == 1 && a.channel == 1) {
		return *this * a(0);
	}
	if (col != a.row) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw errinfo[0];
	}
	if (channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, a.col, channel);
	for (int z = 0; z < channel; z++)
		for (int i = 0; i < row; i++)
			for (int j = 0; j < a.col; j++) {
				float temp = 0;
				for (int d = 0; d < col; d++)
					temp = temp + matrix[(i*col + d)*channel + z] * a((j + d * a.col)*channel + z);
				mark((j + i * a.col)*channel + z) = temp;
			}
	return mark;
}
void Matrix::operator*=(const float val)
{
	*this = *this * val;
}
void Matrix::operator*=(const Matrix & a)
{
	*this = *this * a;
}
const Matrix Matrix::operator / (const float val)const
{
	return (*this) * (1.0f / val);
}
const Matrix Matrix::operator / (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && channel == 1) {
		return (*this)(0) / a;
	}
	else if (a.row == 1 && a.col == 1 && a.channel == 1) {
		return *this / a(0);
	}
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark((j + i * col)*channel + z) = matrix[(j + i * col)*channel + z] / a((j + i * col)*channel + z);
	return mark;
}
void Matrix::operator/=(const float val)
{
	*this = *this / val;
}
void Matrix::operator/=(const Matrix & a)
{
	*this = *this / a;
}
Matrix& Matrix::operator = (const Matrix &temp)
{
	if (this == &temp)
		return *this;
#ifdef LIGHT_MAT
	release();
	recount = temp.recount;
	if (recount != nullptr)
		*recount += 1;
	row = temp.row;
	col = temp.col;
	channel = temp.channel;
	matrix = temp.matrix;
#else
	setvalue(temp);
#endif
	return *this;
}
bool Matrix::operator == (const Matrix &a)const
{
	if (col != a.col) {
		return false;
	}
	if (row != a.row) {
		return false;
	}
	if (channel != a.channel) {
		return false;
	}
	if (memcmp(matrix, a.matrix, col*row*channel * sizeof(float)) == 0)
		return true;
	return false;
}
bool Matrix::operator != (const Matrix & a)const
{
	return !(*this == a);
}
float & Matrix::operator()(const int index) const
{
	if (index > length() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[index];
}
float& Matrix::operator()(const int row, const int col) const
{
	if (row > this->row - 1 || col > this->col - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (row < 0 || col < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[(row*this->col + col)*channel];
}
float & Matrix::operator()(const int row, const int col, const int depth) const
{
	if (row > this->row - 1 || col > this->col - 1 || depth > this->channel - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (row < 0 || col < 0 || depth < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[(row*this->col + col)*this->channel + depth];
}
const Matrix Matrix::operator()(const int index, X_Y_Z rc) const
{
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	switch (rc) {
	case ROW:
		if (index > this->row - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		return Block(*this, index, index, 0, col - 1);
	case COL:
		if (index > this->col - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		return Block(*this, 0, row - 1, index, index);
	case CHANNEL:
		if (index > channel - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		return mSplit(*this, index);
	default:return Matrix();
	}
}
const Matrix nn::Matrix::operator()(const int v1, const int v2, X_Y_Z rc) const
{
	if (v1 < 0 || v2 < 0 ) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix m;
	switch (rc) {
	case ROW:
		if (v1 > col - 1 || v2 > channel - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		m.create(row, 1, 1);
		for (int i = 0; i < row; i++)
			m(i) = matrix[i*col*channel + v1 * channel + v2];
		break;
	case COL:
		if (v1 > row - 1 || v2 > channel - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		m.create(1, col, 1);
		for (int i = 0; i < col; i++)
			m(i) = matrix[v1*col*channel + i * channel + v2];
		break;
	case CHANNEL:
		if (v1 > row - 1 || v2 > col - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		m.create(1, 1, channel);
		for (int i = 0; i < channel; i++)
			m(i) = matrix[v1*col*channel + v2 * channel + i];
		break;
	default:break;
	}
	return m;
}
const Matrix Matrix::operator [] (const int channel)const
{
	return mSplit(*this, channel);
}

const Mat nn::operator + (const float value, const Mat &mat)
{
	return mat + value;
}

const Mat nn::operator - (const float value, const Mat &mat)
{
	return value + (-mat);
}

const Mat nn::operator * (const float value, const Mat &mat)
{
	return mat * value;
}

const Mat nn::operator / (const float value, const Mat &mat)
{
	return Divi(mat, value, LEFT);
}

ostream & nn::operator << (ostream &out, const Mat &ma)
{
	if (ma.matrix == nullptr)
		cout << errinfo[ERR_INFO_EMPTY] << endl;
	else
		ma.show();
	return out;
}
