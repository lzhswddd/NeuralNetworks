#ifndef __MAT_H__
#define __MAT_H__

#include "Matrix.h"
#ifndef DLength
#define DLength(x) (sizeof(x)/sizeof(float))
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
	float Max(const Mat &src, bool isAbs = false);
	/**
	@brief ���ؾ������Сֵ
	@param src ����
	@param isAbs �Ƿ�ȡ����ֵ
	*/
	float Min(const Mat &src, bool isAbs = false);
	/**
	@brief ���ؾ��������ʽ
	@param src ����
	*/
	float det(const Mat &src);
	/**
	@brief ���ؾ���ļ�
	@param src ����
	*/
	float trace(const Mat &src);
	/**
	@brief ���ؾ����Ӧ����������ʽ
	@param src ����
	@param x ������
	@param y ������
	*/
	float cof(const Mat &src, int x, int y);
	/**
	@brief ���������
	@param min ��Сֵ
	@param max ���ֵ
	@param isdouble �Ƿ����������
	*/
	float getRandData(int min, int max, bool isdouble = false);
	/**
	@brief ���ؾ�����
	@param src ����
	@param num ������
	*/
	float mNorm(const Mat& src, int num = 1);
	/**
	@brief ���ؾ���ľ���
	@param a ����
	@param b ����
	@param num ������
	*/
	float mDistance(const Mat& a, const Mat& b, int num = 2);
	/**
	@brief ��������ľ���Ԫ��
	@param src ����
	*/
	float mRandSample(const Mat &src);
	/**
	@brief ���ؽ�����ת����1*point.size()*1�ľ���
	@param point ����
	*/
	const Mat VectoMat(std::vector<float> &point);
	/**
	@brief ���ؽ�����ת����point.size()*point[0].size()*1�ľ���
	@param points ����
	*/
	const Mat VectoMat(std::vector<std::vector<float>> &points);
	/**
	@brief ���ؽ�����ת����һά����
	@param src ����
	*/
	std::vector<float> MattoVec(const Mat &src);
	/**
	@brief ���ؽ�����ת����rowά����
	@param src ����
	*/
	std::vector<std::vector<float>> MattoVecs(const Mat &src);
	/**
	@brief �������ɵ�n*n*1��λ����
	@param n �����С
	*/
	const Mat eye(int n);
	/**
	@brief ���ؾ���ĵ�channel��ͨ��
	@param src ����
	@param channel ͨ������
	*/
	const Mat mSplit(const Mat &src, int channel);
	/**
	@brief ���ذ������ͨ��������
	@param src �������
	@param dst �������ͨ�����ľ�������
	*/
	void mSplit(const Mat &src, Mat *dst);
	/**
	@brief ���ذ�ͨ���ϲ��ľ���
	@param src ��������
	@param channels ͨ����
	*/
	const Mat mMerge(const Mat *src, int channels);
	/**
	@brief ���ذ����������и�ľ���
	@param src ����
	@param Row_Start ��ȡ�г�ʼ����ֵ
	@param Row_End ��ȡ�н�������ֵ
	@param Col_Start ��ȡ�г�ʼ����ֵ
	@param Col_End ��ȡ�н�������ֵ
	*/
	const Mat Block(const Mat &src, int Row_Start, int Row_End, int Col_Start, int Col_End);
	/**
	@brief �����������Ԫ��n*n*1����
	@param n �����С
	@param low �½�
	@param top �Ͻ�
	@param isdouble �Ƿ����ɸ�����
	*/
	const Mat mRand(int low, int top, int n, bool isdouble = false);
	/**
	@brief �����������Ԫ��size.x*size.y*size.z����
	@param low �½�
	@param top �Ͻ�
	@param size �����С
	@param isdouble �Ƿ����ɸ�����
	*/
	const Mat mRand(int low, int top, Size3 size, bool isdouble = false);
	/**
	@brief �����������Ԫ��row*col*channel����
	@param row ��������
	@param col ��������
	@param low �½�
	@param top �Ͻ�
	@param isdouble �Ƿ����ɸ�����
	*/
	const Mat mRand(int low, int top, int row, int col, int channel = 1,bool isdouble = false);
	/**
	@brief ���ش�СΪrow*col*1����
	@param row ��������
	@param col ��������
	*/
	const Mat mcreate(int row, int col);	
	/**
	@brief ���ش�СΪrow*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Mat mcreate(int row, int col, int channel);
	/**
	@brief ���ش�СΪsize����
	@param size �����С
	*/
	const Mat mcreate(Size size);
	/**
	@brief ���ش�СΪsize����
	@param size �����С
	*/
	const Mat mcreate(Size3 size);
	/**
	@brief ����Ԫ��Ϊ0��row*col*1����
	@param row ��������
	@param col ��������
	*/
	const Mat zeros(int row, int col);
	/**
	@brief ����Ԫ��Ϊ0��row*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Mat zeros(int row, int col, int channel);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Mat zeros(Size size);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Mat zeros(Size3 size);
	/**
	@brief ����Ԫ��Ϊv��row*col*channel����
	@param v ���Ԫ��
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Mat value(float v, int row, int col, int channel = 1);
	/**
	@brief ����Ԫ��Ϊ1��row*col*1����
	@param row ��������
	@param col ��������
	@param v ���Ԫ��
	*/
	const Mat ones(int row, int col);
	/**
	@brief ����Ԫ��Ϊ1��row*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Mat ones(int row, int col, int channel);
	/**
	@brief ����Ԫ��Ϊ1��size����
	@param size �����С
	*/
	const Mat ones(Size size);
	/**
	@brief ����Ԫ��Ϊ1��size����
	@param size �����С
	*/
	const Mat ones(Size3 size);
	/**
	@brief ����������󣬾�����Ϊһά����
	@param src ����
	*/
	const Mat reverse(const Mat &src);
	/**
	@brief ��������row*col*channel�ľ��������ȡ����srcԪ�����
	@param src ����
	@param row ��������
	@param col ��������
	*/
	const Mat mRandSample(const Mat &src, int row, int col, int channel = 1);
	/**
	@brief ���������ȡnum�ξ���src���л�����ɵľ���
	@param src ����
	@param rc ��ȡ��ʽ
	@param num ��ȡ����
	*/
	const Mat mRandSample(const Mat &src, X_Y_Z rc, int num = 1);
	/**
	@brief ���ش�low��top�ȷֳɵ�1*len�ľ���
	@param low �½�
	@param top �Ͻ�
	@param len �ȷָ���
	*/
	const Mat linspace(int low, int top, int len);
	/**
	@brief ���ش�low��top�ȷֳɵ�1*len�ľ���
	@param low �½�
	@param top �Ͻ�
	@param len �ȷָ���
	*/
	const Mat linspace(float low, float top, int len);
	/**
	@brief ���ؾ���İ������
	@param src ����
	*/
	const Mat adj(const Mat &src);
	/**
	@brief ���ؾ���������
	@param src ����
	*/
	const Mat inv(const Mat &src);
	/**
	@brief ���ؾ����α�����
	@param src ����
	@param dire α�����ļ��㷽ʽ
	*/
	const Mat pinv(const Mat &src, direction dire = LEFT);
	/**
	@brief ���ؾ����ת�þ���
	@param src ����
	*/
	const Mat tran(const Mat &src);
	/**
	@brief ���ؾ���ľ���ֵ����
	@param src ����
	*/
	const Mat mAbs(const Mat &src);
	/**
	@brief ����angle��2*2����ת����
	@param angle �Ƕ�
	*/
	const Mat Rotate(float angle);
	/**
	@brief ���ؾ���num����
	@param src ����
	@param num ����
	*/
	const Mat POW(const Mat &src, int num);
	/**
	@brief ���ؾ���ȡ��
	@param src ����
	*/
	const Mat mOpp(const Mat &src);
	/**
	@brief ���ؾ����л���֮��
	@param src ����
	@param rc ��͵ķ���
	*/
	const Mat mSum(const Mat &src, X_Y_Z rc);
	/**
	@brief ���ؾ���Ԫ��ȡָ��
	@param src ����
	*/
	const Mat mExp(const Mat &src);
	/**
	@brief ���ؾ���Ԫ��ȡ����
	@param src ����
	*/
	const Mat mLog(const Mat &src);
	/**
	@brief ���ؾ���Ԫ��ȡ����
	@param src ����
	*/
	const Mat mSqrt(const Mat &src);
	/**
	@brief ���ؾ���Ԫ��ȡnum����
	@param src ����
	@param num ����
	*/
	const Mat mPow(const Mat &src, int num);
	/**
	@brief ���ؾ���val/src��Ԫ�س�
	@param src ����
	@param val ����
	*/
	const Mat Divi(const Mat &src, float val, direction dire = RIGHT);
	/**
	@brief ���ؾ������
	@param a ��������
	@param b ������
	@param dire ������ʽ
	*/
	const Mat Divi(const Mat &a, const Mat &b, direction dire = RIGHT);
	/**
	@brief ���ؾ���Ԫ�ضԳ�
	@param a ����
	@param b ����
	*/
	const Mat Mult(const Mat &a, const Mat &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮������ֵ
	@param a �Ƚ�ֵ
	@param b �ȽϾ���
	*/
	const Mat mMax(float a, const Mat &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮������ֵ
	@param a �ȽϾ���
	@param b �ȽϾ���
	*/
	const Mat mMax(const Mat &a, const Mat &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮�����Сֵ
	@param a �Ƚ�ֵ
	@param b �ȽϾ���
	*/
	const Mat mMin(float a, const Mat &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮�����Сֵ
	@param a �ȽϾ���
	@param b �ȽϾ���
	*/
	const Mat mMin(const Mat &a, const Mat &b);	
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
	Size3 mCalSize(const Mat &src, const Mat &kern, Point & anchor, Size strides, int &top, int &bottom, int &left, int &right);
	/**
	@brief mCalSize �������������ŵı߽�
	���ؾ����С
	@param src ���������ߴ�
	@param kern ����˳ߴ�
	@param anchor ���ض�Ӧ���������
	*/
	Size3 mCalSize(Size3 src, Size3 kern, Point &anchor, Size strides);
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
	const Mat mThreshold(const Mat &src, float boundary, float lower, float upper, int boundary2upper = 1);
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
	const Mat copyMakeBorder(const Mat &src, int top, int bottom, int left, int right, BorderTypes borderType = BORDER_CONSTANT, float value = 0.0f);
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
	const Mat Filter2D(const Mat & input, const Mat & kern, Point anchor = Point(-1, -1), const Size & strides = Size(1, 1), bool is_copy_border = true);
	/**
	��С���˷�
	@param x �Ա���
	@param y �����
	*/
	const Mat LeastSquare(const Mat& x, const Mat &y);
	const Mat Reshape(const Mat& src, Size3 size);
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
