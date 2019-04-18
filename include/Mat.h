#ifndef __MAT_H__
#define __MAT_H__

#include "Matrix.h"
#ifndef DLength
#define DLength(x) (sizeof(x)/sizeof(double))
#endif
#ifndef ILength
#define ILength(x) (sizeof(x)/sizeof(int))
#endif
namespace nn
{
	void check(int row, int col, int depth = 1);
	/**
	@brief �������������
	*/
	void Srandom();

	/**
	@brief ���ؾ�������ֵ
	@param src ����
	@param isAbs �Ƿ�ȡ����ֵ
	*/
	double Max(const Matrix &src, bool isAbs = false);
	/**
	@brief ���ؾ������Сֵ
	@param src ����
	@param isAbs �Ƿ�ȡ����ֵ
	*/
	double Min(const Matrix &src, bool isAbs = false);
	/**
	@brief ���ؾ��������ʽ
	@param src ����
	*/
	double det(const Matrix &src);
	/**
	@brief ���ؾ���ļ�
	@param src ����
	*/
	double trace(const Matrix &src);
	/**
	@brief ���ؾ����Ӧ����������ʽ
	@param src ����
	@param x ������
	@param y ������
	*/
	double cof(const Matrix &src, int x, int y);
	/**
	@brief ���������
	@param min ��Сֵ
	@param max ���ֵ
	@param isdouble �Ƿ����������
	*/
	double getRandData(int min, int max, bool isdouble = false);
	/**
	@brief ���ؾ�����
	@param src ����
	@param num ������
	*/
	double mNorm(const Matrix& src, int num = 1);
	/**
	@brief ���ؾ���ľ���
	@param a ����
	@param b ����
	@param num ������
	*/
	double mDistance(const Matrix& a, const Matrix& b, int num = 2);
	/**
	@brief ��������ľ���Ԫ��
	@param src ����
	*/
	double mRandSample(const Matrix &src);
	/**
	@brief ���ؽ�����ת����1*point.size()*1�ľ���
	@param point ����
	*/
	const Matrix VectoMat(std::vector<double> &point);
	/**
	@brief ���ؽ�����ת����point.size()*point[0].size()*1�ľ���
	@param points ����
	*/
	const Matrix VectoMat(std::vector<std::vector<double>> &points);
	/**
	@brief ���ؽ�����ת����һά����
	@param src ����
	*/
	std::vector<double> MattoVec(const Matrix &src);
	/**
	@brief ���ؽ�����ת����rowά����
	@param src ����
	*/
	std::vector<std::vector<double>> MattoVecs(const Matrix &src);
	/**
	@brief �������ɵ�n*n*1��λ����
	@param n �����С
	*/
	const Matrix eye(int n);
	/**
	@brief ���ؾ���ĵ�channel��ͨ��
	@param src ����
	@param channel ͨ������
	*/
	const Matrix mSplit(const Matrix &src, int channel);
	/**
	@brief ���ذ������ͨ��������
	@param src �������
	@param dst �������ͨ�����ľ�������
	*/
	void mSplit(const Matrix &src, Matrix *dst);
	/**
	@brief ���ذ�ͨ���ϲ��ľ���
	@param src ��������
	@param channels ͨ����
	*/
	const Matrix mMerge(const Matrix *src, int channels);
	/**
	@brief ���ذ����������и�ľ���
	@param src ����
	@param Row_Start ��ȡ�г�ʼ����ֵ
	@param Row_End ��ȡ�н�������ֵ
	@param Col_Start ��ȡ�г�ʼ����ֵ
	@param Col_End ��ȡ�н�������ֵ
	*/
	const Matrix Block(const Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End);
	/**
	@brief �����������Ԫ��n*n*1����
	@param n �����С
	@param low �½�
	@param top �Ͻ�
	@param isdouble �Ƿ����ɸ�����
	*/
	const Matrix mRand(int low, int top, int n, bool isdouble = false);
	/**
	@brief �����������Ԫ��row*col*channel����
	@param row ��������
	@param col ��������
	@param low �½�
	@param top �Ͻ�
	@param isdouble �Ƿ����ɸ�����
	*/
	const Matrix mRand(int low, int top, int row, int col, int channel = 1,bool isdouble = false);
	/**
	@brief ���ش�СΪrow*col*1����
	@param row ��������
	@param col ��������
	*/
	const Matrix mcreate(int row, int col);	
	/**
	@brief ���ش�СΪrow*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix mcreate(int row, int col, int channel);
	/**
	@brief ���ش�СΪsize����
	@param size �����С
	*/
	const Matrix mcreate(Size size);
	/**
	@brief ���ش�СΪsize����
	@param size �����С
	*/
	const Matrix mcreate(Size3 size);
	/**
	@brief ����Ԫ��Ϊ0��row*col*1����
	@param row ��������
	@param col ��������
	*/
	const Matrix zeros(int row, int col);
	/**
	@brief ����Ԫ��Ϊ0��row*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix zeros(int row, int col, int channel);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Matrix zeros(Size size);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Matrix zeros(Size3 size);
	/**
	@brief ����Ԫ��Ϊv��row*col*channel����
	@param v ���Ԫ��
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix value(double v, int row, int col, int channel = 1);
	/**
	@brief ����Ԫ��Ϊ1��row*col*1����
	@param row ��������
	@param col ��������
	@param v ���Ԫ��
	*/
	const Matrix ones(int row, int col);
	/**
	@brief ����Ԫ��Ϊ0��row*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix ones(int row, int col, int channel);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Matrix ones(Size size);
	/**
	@brief ����������󣬾�����Ϊһά����
	@param src ����
	*/
	const Matrix reverse(const Matrix &src);
	/**
	@brief ��������row*col*channel�ľ��������ȡ����srcԪ�����
	@param src ����
	@param row ��������
	@param col ��������
	*/
	const Matrix mRandSample(const Matrix &src, int row, int col, int channel = 1);
	/**
	@brief ���������ȡnum�ξ���src���л�����ɵľ���
	@param src ����
	@param rc ��ȡ��ʽ
	@param num ��ȡ����
	*/
	const Matrix mRandSample(const Matrix &src, X_Y_Z rc, int num = 1);
	/**
	@brief ���ش�low��top�ȷֳɵ�1*len�ľ���
	@param low �½�
	@param top �Ͻ�
	@param len �ȷָ���
	*/
	const Matrix linspace(int low, int top, int len);
	/**
	@brief ���ش�low��top�ȷֳɵ�1*len�ľ���
	@param low �½�
	@param top �Ͻ�
	@param len �ȷָ���
	*/
	const Matrix linspace(double low, double top, int len);
	/**
	@brief ���ؾ���İ������
	@param src ����
	*/
	const Matrix adj(const Matrix &src);
	/**
	@brief ���ؾ���������
	@param src ����
	*/
	const Matrix inv(const Matrix &src);
	/**
	@brief ���ؾ����α�����
	@param src ����
	@param dire α�����ļ��㷽ʽ
	*/
	const Matrix pinv(const Matrix &src, direction dire = LEFT);
	/**
	@brief ���ؾ����ת�þ���
	@param src ����
	*/
	const Matrix tran(const Matrix &src);
	/**
	@brief ���ؾ���ľ���ֵ����
	@param src ����
	*/
	const Matrix mAbs(const Matrix &src);
	/**
	@brief ����angle��2*2����ת����
	@param angle �Ƕ�
	*/
	const Matrix Rotate(double angle);
	/**
	@brief ���ؾ���num����
	@param src ����
	@param num ����
	*/
	const Matrix POW(const Matrix &src, int num);
	/**
	@brief ���ؾ���ȡ��
	@param src ����
	*/
	const Matrix mOpp(const Matrix &src);
	/**
	@brief ���ؾ����л���֮��
	@param src ����
	@param rc ��͵ķ���
	*/
	const Matrix mSum(const Matrix &src, X_Y_Z rc);
	/**
	@brief ���ؾ���Ԫ��ȡָ��
	@param src ����
	*/
	const Matrix mExp(const Matrix &src);
	/**
	@brief ���ؾ���Ԫ��ȡ����
	@param src ����
	*/
	const Matrix mLog(const Matrix &src);
	/**
	@brief ���ؾ���Ԫ��ȡ����
	@param src ����
	*/
	const Matrix mSqrt(const Matrix &src);
	/**
	@brief ���ؾ���Ԫ��ȡnum����
	@param src ����
	@param num ����
	*/
	const Matrix mPow(const Matrix &src, int num);
	/**
	@brief ���ؾ���val/src��Ԫ�س�
	@param src ����
	@param val ����
	*/
	const Matrix Divi(const Matrix &src, double val, direction dire = RIGHT);
	/**
	@brief ���ؾ������
	@param a ��������
	@param b ������
	@param dire ������ʽ
	*/
	const Matrix Divi(const Matrix &a, const Matrix &b, direction dire = RIGHT);
	/**
	@brief ���ؾ���Ԫ�ضԳ�
	@param a ����
	@param b ����
	*/
	const Matrix Mult(const Matrix &a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮������ֵ
	@param a �Ƚ�ֵ
	@param b �ȽϾ���
	*/
	const Matrix mMax(double a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮������ֵ
	@param a �ȽϾ���
	@param b �ȽϾ���
	*/
	const Matrix mMax(const Matrix &a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮�����Сֵ
	@param a �Ƚ�ֵ
	@param b �ȽϾ���
	*/
	const Matrix mMin(double a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮�����Сֵ
	@param a �ȽϾ���
	@param b �ȽϾ���
	*/
	const Matrix mMin(const Matrix &a, const Matrix &b);	
	/**
	@brief mCalSize �������������ŵı߽�
	���ؾ����С
	@param src ���������
	@param kern �����
	@param anchor ���ض�Ӧ���������
	anchorĬ��ΪPoint(-1,-1), ���ض�Ӧ���������
	@param strides ��������
	@param top �������伸��
	@param bottom �������伸��
	@param left �������伸��
	@param right �������伸��
	*/
	Size3 mCalSize(const Matrix &src, const Matrix &kern, Point & anchor, Size strides, int &top, int &bottom, int &left, int &right);
	/**
	@brief mCalSize �������������ŵı߽�
	���ؾ����С
	@param src ���������ߴ�
	@param kern ����˳ߴ�
	@param anchor ���ض�Ӧ���������
	*/
	Size3 mCalSize(Size3 src, Size3 kern, Point & anchor, Size strides);
	/**
	@brief ���ذ�boundary�ֽ����ľ���
	���ؾ����С������������С
	@param src �������
	@param boundary �ֽ�ֵ
	@param lower С��boundary��lower���
	@param upper ����boundary��upper���
	@param boundary2upper ������Ԫ�ص���boundaryʱ
	Ϊ1��upper, Ϊ-1��lower, Ϊ0������
	*/
	const Matrix mThreshold(const Matrix &src, double boundary, double lower, double upper, int boundary2upper = 1);
	/**
	@brief ���ر߽�����ľ���
	@param src �������
	@param top �������伸��
	@param bottom �������伸��
	@param left �������伸��
	@param right �������伸��
	@param borderType �߽��������ƵĲ�ֵ����
	@param value ������ֵ����ֵ
	**/
	const Matrix copyMakeBorder(const Matrix &src, int top, int bottom, int left, int right, BorderTypes borderType = BORDER_CONSTANT, const int value = 0);
	/**
	@brief ���ؾ���2ά������
	���ؾ����СΪ(input.row/strides_x, input.col/strides_y, 1)
	@param input �������
	@param kern �����
	@param anchor ����Ԫ�ض�Ӧ����˵�λ��
	�Ծ���˵����Ͻ�Ϊ(0,0)��, Ĭ��(-1,-1)Ϊ����
	@param strides �������� 
	Size.heiΪx��,Size.widΪy��
	@param is_copy_border �Ƿ�Ҫ��չ�߽�
	*/
	const Matrix Filter2D(const Mat & input, const Mat & kern, Point anchor = Point(-1, -1), const Size & strides = Size(1, 1), bool is_copy_border = true);
	/**
	��С���˷�
	@param x �Ա���
	@param y �����
	*/
	const Matrix LeastSquare(const Matrix& x, const Matrix &y);
	/**
	@brief �����а��������
	@param row ��
	@param col ��
	*/
	template<typename T>
	void showMatrix(const T *, int row, int col); 
	void pause();
}

#endif //__MAT_H__
