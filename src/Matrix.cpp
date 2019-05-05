#include <iostream>
#include <fstream>
#include <iomanip>
#include "alignmalloc.h"
#include "function.h"
#include "mat.h"
#include "matrix.h"
using namespace std;
using namespace nn;

#ifdef MAT_DEBUG
#define CHECK_MATRIX(matrix) if((matrix) == nullptr) {cerr << errinfo[ERR_INFO_EMPTY] << endl;throw std::exception(errinfo[0]);}
#endif // MAT_DEBUG

Matrix::Matrix()
{
	init();
	checkSquare();
}
Matrix::Matrix(int row, int col)
{
	init();
	check(row, col);
	create(row, col, 1);
}
Matrix::Matrix(int row, int col, int depth)
{
	init();
	check(row, col);
	create(row, col, depth);
}
Matrix::Matrix(Size size_)
{
	init();
	check(size_.h, size_.w);
	create(size_.h, size_.w, 1);
}
Matrix::Matrix(Size3 size_)
{
	init();
	check(size_.h, size_.w, size_.c);
	create(size_.h, size_.w, size_.c);
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
		setsize(row, col, channel);
#ifdef LIGHT_MAT
		createCount();
#endif
		this->matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
		if (this->matrix != nullptr)
			for (int index = 0; index < length(); index++)
				(*this)(index) = (float)matrix[index];
	}
	checkSquare();
}
Matrix::Matrix(float *matrix, int row, int col, int channel)
{
	init();
	if (matrix != nullptr) {
		check(row, col);
		setsize(row, col, channel);
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
	setsize(1, w, 1);
	matrix = data;
}
Matrix::Matrix(int w, int h, float * data)
{
	init();
	setsize(h, w, 1);
	matrix = data;
}
Matrix::Matrix(int w, int h, int c, float * data)
{
	init();
	setsize(h, w, c);
	matrix = data;
}
nn::Matrix::Matrix(int w, int h, int c, int c_offset, float * data)
{
	init();
	setsize(h, w, c);
	offset_c = c_offset;
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
Matrix::Matrix(const Matrix &a, const Matrix &b, X_Y_Z merge)
{
	init();
	if (merge == ROW) {
		if (a.col == b.col) {
			create(a.row + b.row, a.col, a.channel);
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
			create(a.row, a.col + b.col, a.channel);
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
	setsize(0, 0, 0);
	/*if (matrix != nullptr) {
		fastFree(matrix);
		matrix = nullptr;
	}*/
}

void Matrix::create(int w)
{
	release();
	check(w, 1, 1);
	setsize(1, w, 1);
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
	setsize(h, w, 1);
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
	setsize(h, w, c);
#ifdef LIGHT_MAT
	createCount();
#endif
	matrix = (float*)fastMalloc(row*col*channel * sizeof(float));
	checkSquare();
}

void nn::Matrix::create(Size size)
{
	create(size.h, size.w);
}

void nn::Matrix::create(Size3 size)
{
	create(size.h, size.w, size.c);
}

float* Matrix::mat_()const
{
	return matrix;
}

void Matrix::DimCheck()const
{
	if (channel != 1) {
		cerr << errinfo[ERR_INFO_DIM] << endl;
		throw std::exception(errinfo[ERR_INFO_DIM]);
	}
}

void Matrix::copy(Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	int hei = Row_End - Row_Start + 1;
	int wid = Col_End - Col_Start + 1;
	check(hei, wid);
	if (src.matrix == nullptr) {
		src = zeros(hei, wid, channel);
	}
	for (int y = Row_Start, j = 0; y <= Row_End; y++, j++)
		for (int x = Col_Start, i = 0; x <= Col_End; x++, i++)
			for (int z = 0; z < channel; z++)
				src(y, x, z) = (*this)(j, i, z);
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
						temp(i, j, z) = (*this)(i, j - 1, z);
				}
				else if (dire == RIGHT) {
					if (j == col)
						temp(i, j, z) = 1;
					else
						temp(i, j, z) = (*this)(i, j - 1, z); 
				}
			}
		}
	}
	*this = temp;
}
void Matrix::mChannel(const Matrix & src, int channels)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	CHECK_MATRIX(src.matrix);
	if (row != src.row || col != src.col || channels >= channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			(*this)(i, j, channels) = src(i, j);
		}
	}
}
void nn::Matrix::mChannel(const Matrix & src, int row, int col)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	CHECK_MATRIX(src.matrix);
	if (this->row <= row || this->col <= col || src.channels() != channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	for (int i = 0; i < channel; ++i) {
		(*this)(row, col, i) = src(i);
	}
}
void nn::Matrix::reshape(Size3 size)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (length() != size.h * size.w*size.c) {
		fprintf(stderr, errinfo[ERR_INFO_UNLESS]);
		throw std::exception(errinfo[ERR_INFO_UNLESS]);
	}
#endif // MAT_DEBUG
	setsize(size.h, size.w, size.c);
}
void Matrix::reshape(int row, int col, int channels)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (length() != row * col*channels) {
		fprintf(stderr, errinfo[ERR_INFO_UNLESS]);
		throw std::exception(errinfo[ERR_INFO_UNLESS]);
	}
#endif // MAT_DEBUG
	setsize(row, col, channels);
}
bool Matrix::setSize(int row, int col, int channel)
{
	if (row*col*channel > 0) {
		*this = Matrix(row, col, channel);
		return true;
	}
	if (length() == row * col * channel) {
		setsize(row, col, channel);
		return true;
	}
	return false;
}
void Matrix::setNum(float number, int index)
{
#ifdef MAT_DEBUG
	checkindex(index);
#endif // MAT_DEBUG
	(*this)(index) = number;
}
void Matrix::setNum(float number, int index_y, int index_x)
{
#ifdef MAT_DEBUG
	checkindex(index_x, index_y);
#endif // MAT_DEBUG
	(*this)(index_y, index_x) = number;
}
void Matrix::setMat(float *mat, int hei, int wid)
{
	if ((row <= 0 || col <= 0)) return;
	*this = Matrix(mat, hei, wid);
	checkSquare();
}
void Matrix::setvalue(const Matrix &src)
{
	setsize(src.row, src.col, src.channel);
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
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (isEnable() == 0)
		*this = adj();
	else {
		cerr << errinfo[ERR_INFO_ADJ] << endl;
		throw std::exception(errinfo[ERR_INFO_ADJ]);
	}
}
void Matrix::setTran()
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	*this = t();
}
void Matrix::setInv()
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (isEnable() == 0)
		*this = inv();
	else {
		cerr << errinfo[ERR_INFO_INV] << endl;
		throw std::exception(errinfo[ERR_INFO_INV]);
	}
}
void Matrix::setPow(int num)
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (isEnable() == 0)
		*this = pow(num);
	else {
		cerr << errinfo[ERR_INFO_POW] << endl;
		throw std::exception(errinfo[ERR_INFO_POW]);
	}
}
void Matrix::setIden()
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (isEnable() == 0)
		*this = eye(row);
	else {
		cerr << errinfo[ERR_INFO_IND] << endl;
		throw std::exception(errinfo[ERR_INFO_IND]);
	}
}
Size3 Matrix::size3() const
{
	return Size3(row, col, channel);
}
int nn::Matrix::total() const
{
	return offset_c;
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
							out << setw(8) << setprecision(2) << setfill(' ') << (*this)(i, j, k) << " ]]";
						}
						else if (k == channel - 1)
						{
							out << setw(8) << setprecision(2) << setfill(' ') << (*this)(i, j, k) << " ]";
						}
						else {
							out << setw(8) << setprecision(2) << setfill(' ') << (*this)(i, j, k) << ", ";
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
void Matrix::copyTo(const Matrix & mat) const
{
#ifdef MAT_DEBUG
	if (matrix == nullptr || mat.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
	if (row != mat.row || col != mat.col || channel != mat.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif //MAT_DEBUG
	for (int i = 0; i < length(); i++)
		mat(i) = (*this)(i);
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
	return (*this)(index_y, index_x);
}
float& Matrix::at(int index)const
{
	return (*this)(index);
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
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	return (*this)(0);
}
float& Matrix::findAt(float value)const
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	static float err = NAN;
	for (int ind = 0; ind < length(); ind++)
		if ((*this)(ind) == value)
			return (*this)(ind);
	return err;
}
float& Matrix::findmax()const
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	int max_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if ((*this)(max_adr) < (*this)(ind))
			max_adr = ind;
	return (*this)(max_adr);
}
float& Matrix::findmin()const
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	int min_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if ((*this)(min_adr) < (*this)(ind))
			min_adr = ind;
	return (*this)(min_adr);
}
int Matrix::find(float value)const
{
#ifdef MAT_DEBUG
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	for (int ind = 0; ind < length(); ind++)
		if ((*this)(ind) == value)
			return ind;
	return -1;
}
int Matrix::maxAt()const
{
#ifdef MAT_DEBUG
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	int max_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if ((*this)(max_adr) < (*this)(ind))
			max_adr = ind;
	return max_adr;
}
int Matrix::minAt()const
{
#ifdef MAT_DEBUG
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	int min_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if ((*this)(min_adr) < (*this)(ind))
			min_adr = ind;
	return min_adr;
}
bool Matrix::contains(float value)const
{
#ifdef MAT_DEBUG
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	for (int ind = 0; ind < length(); ind++)
		if ((*this)(ind) == value)
			return true;
	return false;
}

void Matrix::show()const
{
#ifdef MAT_DEBUG
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	cout.setf(ios::scientific);
	cout.setf(ios::showpos);
	cout.setf(ios::left);
	for (int z = 0; z < channel; z++) {
		for (int i = 0; i < row; i++) {
			cout << "[ ";
			for (int j = 0; j < col - 1; j++) {
				cout << setw(8) << setprecision(2) << setfill(' ') << (*this)(i, j, z) << ", ";
			}
			cout << setw(8) << setprecision(2) << setfill(' ') << (*this)(i, col - 1, z);
			cout << " ]";
		}
	}
	cout.unsetf(ios::scientific);
	cout.unsetf(ios::showpos);
	cout.unsetf(ios::left);
	cout << defaultfloat << setprecision(6);
}

const Matrix Matrix::abs()const
{
	return mAbs(*this);
}
const Matrix nn::Matrix::mPow(int num) const
{
#ifdef MAT_DEBUG
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (square) {
		return POW(*this, num);
	}
	else{
		cerr << errinfo[ERR_INFO_POW] << endl;
		throw std::exception(errinfo[ERR_INFO_POW]);
	}
}
const Matrix Matrix::pow(float num)const
{
	return nn::mPow(*this, num);
}
float Matrix::sum(int num, bool _abs)const
{
#ifdef MAT_DEBUG
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	float sum = 0;
	if (num == 0) {
		return (float)length();
	}
	else if (num == 1) {
		for (int ind = 0; ind < length(); ind++)
			if (_abs)
				sum += fabs((*this)(ind));
			else
				sum += (*this)(ind);
	}
	else
		for (int ind = 0; ind < length(); ind++)
			if (_abs)
				sum += std::pow(fabs((*this)(ind)), num);
			else
				sum += std::pow((*this)(ind), num);
	return sum;
}

float nn::Matrix::mean() const
{
	float sum = 0;
	for (int ind = 0; ind < length(); ind++)
		sum += (*this)(ind);
	return sum / (float)length();
}

Matrix nn::Matrix::Channel(int c)
{
#ifdef MAT_DEBUG
	if (c < 0) {
		cerr << errinfo[ERR_INFO_UNLESS] << endl;
		throw std::exception(errinfo[ERR_INFO_UNLESS]);
	}
	if (c >= channel) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
#endif // DEBUG_MAT
	return Matrix(row, col, 1, channel, matrix + c);
}
const Matrix nn::Matrix::Channel(int c) const
{
#ifdef MAT_DEBUG
	if (c < 0) {
		cerr << errinfo[ERR_INFO_UNLESS] << endl;
		throw std::exception(errinfo[ERR_INFO_UNLESS]);
	}
	if (c >= channel) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
#endif // DEBUG_MAT
	return Matrix(row, col, 1, channel, matrix + c);
}
const Matrix nn::Matrix::clone() const
{
	Matrix dst;
	dst.setvalue(*this);
	return dst;
}
const Matrix Matrix::opp()const
{
	return mOpp(*this);
}
const Matrix Matrix::adj()const
{
#ifdef MAT_DEBUG
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (square) {
		return nn::adj(*this);
	}
	else {
		cerr << errinfo[ERR_INFO_ADJ] << endl;
		throw std::exception(errinfo[ERR_INFO_ADJ]);
	}
}
const Matrix Matrix::t()const
{
#ifdef MAT_DEBUG
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	return tran(*this);
}
const Matrix Matrix::inv()const
{
#ifdef MAT_DEBUG
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
#endif //MAT_DEBUG
	if (square) {
		return nn::inv(*this);
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
				throw std::exception(errinfo[ERR_INFO_PINV]);
			}
		}
	}
}
const Matrix Matrix::reverse()const
{
	return nn::reverse(*this);
}
float Matrix::Det()
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	if (square)
		return det(*this);
	else
		return NAN;
}
float Matrix::Norm(int num)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (col != 1 && row != 1) {
		cerr << errinfo[ERR_INFO_NORM] << endl;
		throw std::exception(errinfo[ERR_INFO_NORM]);
	}
	if (num < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
#endif // MAT_DEBUG
	if (num == 0)
		return sum();
	else if (num == 1)
		return sum(1, true);
	else if (num == 2)
		return std::sqrt(sum(2, true));
	//else if (isinf(num) == 1)
	//	return abs(matrix[find(findmax())]);
	//else if (isinf(num) == -1)
	//	return abs(matrix[find(findmin())]);
	else
		return std::pow(sum(num, true), 1 / float(num));
}
float Matrix::Matrix::Cof(int x, int y)
{
	return cof(*this, x, y);
}
float Matrix::EigenvalueMax(float offset)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	if (square) {
		int count = 0;
		float err = 100 * offset;
		Matrix v;
		Matrix u0 = ones(row, 1);
		while (err > offset) {
			v = *this*u0;
			Matrix u1 = v * (1 / v.findmax());
			err = (u1 - u0).abs().findmax();
			u0 = u1; count += 1;
			if (count >= 1e+3) {
				cerr << errinfo[ERR_INFO_EIGEN] << endl;
				throw std::exception(errinfo[ERR_INFO_EIGEN]);
			}
		}
		return v.findmax();
	}
	else {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw std::exception(errinfo[ERR_INFO_SQUARE]);
	}
}
float Matrix::RandSample()
{
	return mRandSample(*this);
}
const Matrix Matrix::EigenvectorsMax(float offset)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	if (square) {
		int count = 0;
		float err = 100 * offset;
		Matrix v;
		Matrix u0 = ones(row, 1);
		while (err > offset) {
			v = *this*u0;
			Matrix u1 = v * (1 / v.findmax());
			err = (u1 - u0).abs().findmax();
			u0 = u1; count += 1;
			if (count >= 1e+3) {
				cerr << errinfo[ERR_INFO_EIGEN] << endl;
				throw std::exception(errinfo[ERR_INFO_EIGEN]);
			}
		}
		return u0;
	}
	else {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw std::exception(errinfo[ERR_INFO_SQUARE]);
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

const Matrix Matrix::exp()const
{
	return mExp(*this);
}

const Matrix Matrix::log()const
{
	return mLog(*this);
}

const Matrix Matrix::sqrt()const
{
	return mSqrt(*this);
}

void Matrix::init()
{
	row = 0;
	col = 0;
	channel = 0;
	matrix = nullptr;
	offset_c = 1;
#ifdef LIGHT_MAT
	recount = nullptr;
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
#ifdef MAT_DEBUG
void Matrix::checkindex(int index)
{
	if (row == 0 || col == 0) {
		cerr << errinfo[ERR_INFO_LEN] << endl;
		throw std::exception(errinfo[ERR_INFO_LEN]);
	}
	if (index > length() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
}
void Matrix::checkindex(int index_x, int index_y)
{
	if (row == 0 || col == 0) {
		cerr << errinfo[ERR_INFO_LEN] << endl;
		throw std::exception(errinfo[ERR_INFO_LEN]);
	}
	if (index_x < 0 || index_y < 0) {
		cerr << errinfo[ERR_INFO_UNLESS] << endl;
		throw std::exception(errinfo[ERR_INFO_UNLESS]);
	}
	if (index_x*col + index_y > row*col - 1 || index_x >= row || index_y >= col) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw std::exception(errinfo[ERR_INFO_EMPTY]);
	}
}
#endif // MAT_DEBUG
void nn::Matrix::setsize(int h, int w, int c)
{
	col = w;
	row = h;
	channel = c;
	offset_c = c;
}

const Matrix Matrix::operator + (const float val)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	Matrix mark(row, col, channel);
	for (int i = 0; i < length(); i++)
		mark(i) = (*this)(i) + val;
	return mark;
}
const Matrix Matrix::operator + (const Matrix &a)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
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
					mat(i, j, z) = (*this)(i, j, z) + a(z);
		return mat;
	}
#ifdef MAT_DEBUG
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	Matrix mark(row, col, channel);
	for (int i = 0; i < length(); i++)
		mark(i) = (*this)(i) + a(i);
	return mark;
}
void Matrix::operator += (const float val)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) += val;
}
void Matrix::operator += (const Matrix & a)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) += a(i);
}
const Matrix Matrix::operator-(void) const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	Matrix mark(row, col, channel);
	for (int i = 0; i < length(); i++)
		mark(i) = -(*this)(i);
	return mark;
}
const Matrix Matrix::operator - (const float val)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	Matrix mark(row, col, channel);
	for (int i = 0; i < length(); i++)
		mark(i) = (*this)(i) - val;
	return mark;
}
const Matrix Matrix::operator - (const Matrix &a)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
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
					mat(i, j, z) = (*this)(i, j, z) - a(z);
		return mat;
	}
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
	Matrix mark(row, col, channel);
	for (int i = 0; i < length(); i++)
		mark(i) = (*this)(i) - a(i);
	return mark;
}
void Matrix::operator-=(const float val)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) -= val;
}
void Matrix::operator-=(const Matrix & a)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) -= a(i);
}
const Matrix Matrix::operator * (const float val)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	Matrix mark(row, col, channel);	
	for (int i = 0; i < length(); i++)
		mark(i) = (*this)(i) * val;
	return mark;
}
const Matrix Matrix::operator * (const Matrix &a)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	CHECK_MATRIX(a.matrix);
#endif // MAT_DEBUG
	if (row == 1 && col == 1 && channel == 1) {
		return (*this)(0) * a;
	}
	else if (a.row == 1 && a.col == 1 && a.channel == 1) {
		return *this * a(0);
	}
#ifdef MAT_DEBUG
	if (col != a.row) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw std::exception(errinfo[ERR_INFO_MULT]);
	}
	if (channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	Matrix mark(row, a.col, channel);
	for (int z = 0; z < channel; z++)
		for (int i = 0; i < row; i++)
			for (int j = 0; j < a.col; j++) {
				float temp = 0;
				for (int d = 0; d < col; d++)
					temp = temp + (*this)(i, d, z) * a(d, j, z);
				mark(i, j, z) = temp;
			}
	return mark;
}
void Matrix::operator*=(const float val)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) *= val;
}
void Matrix::operator*=(const Matrix & a)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) *= a(i);
}
const Matrix Matrix::operator / (const float val)const
{
	return (*this) * (1.0f / val);
}
const Matrix Matrix::operator / (const Matrix &a)const
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	CHECK_MATRIX(a.matrix);
#endif // MAT_DEBUG
	if (row == 1 && col == 1 && channel == 1) {
		return (*this)(0) / a;
	}
	else if (a.row == 1 && a.col == 1 && a.channel == 1) {
		return *this / a(0);
	}
	else if (a.row == 1 && a.col == 1 && a.channel == channel) {
		Matrix mat(row, col, channel);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				for (int z = 0; z < channel; z++)
					mat(i, j, z) = (*this)(i, j, z) / a(z);
		return mat;
	}
#ifdef MAT_DEBUG
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	Matrix mark(row, col, channel);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < channel; z++)
				mark(i, j, z) = (*this)(i, j, z) / a(i, j, z);
	return mark;
}
void Matrix::operator/=(const float val)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) /= val;
}
void Matrix::operator/=(const Matrix & a)
{
#ifdef MAT_DEBUG
	CHECK_MATRIX(matrix);
	if (row != a.row || col != a.col || channel != a.channel) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw std::exception(errinfo[ERR_INFO_SIZE]);
	}
#endif // MAT_DEBUG
	for (int i = 0; i < length(); i++)
		(*this)(i) = (*this)(i) / a(i);
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
	offset_c = temp.offset_c;
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
#ifdef MAT_DEBUG
	if (index > length() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	return matrix[index * offset_c / channel];
}
float& Matrix::operator()(const int row, const int col) const
{
#ifdef MAT_DEBUG
	if (row > this->row - 1 || col > this->col - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (row < 0 || col < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	return matrix[(row*this->col + col)*offset_c];
}
float & Matrix::operator()(const int row, const int col, const int depth) const
{
#ifdef MAT_DEBUG
	if (row > this->row - 1 || col > this->col - 1 || depth > this->channel - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw std::exception(errinfo[ERR_INFO_MEMOUT]);
	}
	if (row < 0 || col < 0 || depth < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	return matrix[(row*this->col + col)*offset_c + depth];
}
const Matrix Matrix::operator()(const int index, X_Y_Z rc) const
{
#ifdef MAT_DEBUG
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	switch (rc) {
	case ROW:
		if (index > this->row - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
		return Block(*this, index, index, 0, col - 1);
	case COL:
		if (index > this->col - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
		return Block(*this, 0, row - 1, index, index);
	case CHANNEL:
		if (index > channel - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
		return mSplit(*this, index);
	default:return Matrix();
	}
}
const Matrix nn::Matrix::operator()(const int v1, const int v2, X_Y_Z rc) const
{
#ifdef MAT_DEBUG
	if (v1 < 0 || v2 < 0 ) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw std::exception(errinfo[ERR_INFO_VALUE]);
	}
	CHECK_MATRIX(matrix);
#endif // MAT_DEBUG
	Matrix m;
	switch (rc) {
	case ROW:
#ifdef MAT_DEBUG
		if (v1 > col - 1 || v2 > channel - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
#endif // MAT_DEBUG
		m.create(row, 1, 1);
		for (int i = 0; i < row; i++)
			m(i) = (*this)(i, v1, v2);
		break;
	case COL:
#ifdef MAT_DEBUG
		if (v1 > row - 1 || v2 > channel - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
#endif // MAT_DEBUG
		m.create(1, col, 1);
		for (int i = 0; i < col; i++)
			m(i) = (*this)(v1, i, v2);
		break;
	case CHANNEL:
#ifdef MAT_DEBUG
		if (v1 > row - 1 || v2 > col - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
#endif // MAT_DEBUG
		return Matrix(1, 1, channel, matrix + (v1*col + v2)*offset_c);
		break;
	default:break;
	}
	return m;
}
const Matrix Matrix::operator [] (const int channel)const
{
	return Channel(channel);
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
#ifdef MAT_DEBUG
	CHECK_MATRIX(ma.matrix);
#endif // MAT_DEBUG
	ma.show();
	return out;
}
