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
	@brief 设置随机数种子
	*/
	void Srandom();

	/**
	@brief 返回矩阵的最大值
	@param src 矩阵
	@param isAbs 是否取绝对值
	*/
	double Max(const Matrix &src, bool isAbs = false);
	/**
	@brief 返回矩阵的最小值
	@param src 矩阵
	@param isAbs 是否取绝对值
	*/
	double Min(const Matrix &src, bool isAbs = false);
	/**
	@brief 返回矩阵的行列式
	@param src 矩阵
	*/
	double det(const Matrix &src);
	/**
	@brief 返回矩阵的迹
	@param src 矩阵
	*/
	double trace(const Matrix &src);
	/**
	@brief 返回矩阵对应索引的余子式
	@param src 矩阵
	@param x 列索引
	@param y 行索引
	*/
	double cof(const Matrix &src, int x, int y);
	/**
	@brief 返回随机数
	@param min 最小值
	@param max 最大值
	@param isdouble 是否随机非整数
	*/
	double getRandData(int min, int max, bool isdouble = false);
	/**
	@brief 返回矩阵范数
	@param src 矩阵
	@param num 几范数
	*/
	double mNorm(const Matrix& src, int num = 1);
	/**
	@brief 返回矩阵的距离
	@param a 矩阵
	@param b 矩阵
	@param num 几范数
	*/
	double mDistance(const Matrix& a, const Matrix& b, int num = 2);
	/**
	@brief 返回随机的矩阵元素
	@param src 矩阵
	*/
	double mRandSample(const Matrix &src);
	/**
	@brief 返回将向量转换成1*point.size()*1的矩阵
	@param point 向量
	*/
	const Matrix VectoMat(std::vector<double> &point);
	/**
	@brief 返回将向量转换成point.size()*point[0].size()*1的矩阵
	@param points 向量
	*/
	const Matrix VectoMat(std::vector<std::vector<double>> &points);
	/**
	@brief 返回将矩阵转换成一维向量
	@param src 矩阵
	*/
	std::vector<double> MattoVec(const Matrix &src);
	/**
	@brief 返回将矩阵转换成row维向量
	@param src 矩阵
	*/
	std::vector<std::vector<double>> MattoVecs(const Matrix &src);
	/**
	@brief 返回生成的n*n*1单位矩阵
	@param n 矩阵大小
	*/
	const Matrix eye(int n);
	/**
	@brief 返回矩阵的第channel的通道
	@param src 矩阵
	@param channel 通道索引
	*/
	const Matrix mSplit(const Matrix &src, int channel);
	/**
	@brief 返回按矩阵的通道数复制
	@param src 输入矩阵
	@param dst 输入矩阵通道数的矩阵数组
	*/
	void mSplit(const Matrix &src, Matrix *dst);
	/**
	@brief 返回按通道合并的矩阵
	@param src 矩阵数组
	@param channels 通道数
	*/
	const Matrix mMerge(const Matrix *src, int channels);
	/**
	@brief 返回按索引区域切割的矩阵
	@param src 矩阵
	@param Row_Start 截取行初始索引值
	@param Row_End 截取行结束索引值
	@param Col_Start 截取列初始索引值
	@param Col_End 截取列结束索引值
	*/
	const Matrix Block(const Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End);
	/**
	@brief 返回随机生成元素n*n*1矩阵
	@param n 矩阵大小
	@param low 下界
	@param top 上界
	@param isdouble 是否生成浮点数
	*/
	const Matrix mRand(int low, int top, int n, bool isdouble = false);
	/**
	@brief 返回随机生成元素row*col*channel矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	@param low 下界
	@param top 上界
	@param isdouble 是否生成浮点数
	*/
	const Matrix mRand(int low, int top, int row, int col, int channel = 1,bool isdouble = false);
	/**
	@brief 返回大小为row*col*1矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	*/
	const Matrix mcreate(int row, int col);	
	/**
	@brief 返回大小为row*col*channel矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel 矩阵通道数
	*/
	const Matrix mcreate(int row, int col, int channel);
	/**
	@brief 返回大小为size矩阵
	@param size 矩阵大小
	*/
	const Matrix mcreate(Size size);
	/**
	@brief 返回大小为size矩阵
	@param size 矩阵大小
	*/
	const Matrix mcreate(Size3 size);
	/**
	@brief 返回元素为0的row*col*1矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	*/
	const Matrix zeros(int row, int col);
	/**
	@brief 返回元素为0的row*col*channel矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel 矩阵通道数
	*/
	const Matrix zeros(int row, int col, int channel);
	/**
	@brief 返回元素为0的size矩阵
	@param size 矩阵大小
	*/
	const Matrix zeros(Size size);
	/**
	@brief 返回元素为0的size矩阵
	@param size 矩阵大小
	*/
	const Matrix zeros(Size3 size);
	/**
	@brief 返回元素为v的row*col*channel矩阵
	@param v 填充元素
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel 矩阵通道数
	*/
	const Matrix value(double v, int row, int col, int channel = 1);
	/**
	@brief 返回元素为1的row*col*1矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	@param v 填充元素
	*/
	const Matrix ones(int row, int col);
	/**
	@brief 返回元素为0的row*col*channel矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel 矩阵通道数
	*/
	const Matrix ones(int row, int col, int channel);
	/**
	@brief 返回元素为0的size矩阵
	@param size 矩阵大小
	*/
	const Matrix ones(Size size);
	/**
	@brief 返回逆序矩阵，矩阵需为一维向量
	@param src 矩阵
	*/
	const Matrix reverse(const Matrix &src);
	/**
	@brief 返回生成row*col*channel的矩阵，随机抽取矩阵src元素填充
	@param src 矩阵
	@param row 矩阵行数
	@param col 矩阵列数
	*/
	const Matrix mRandSample(const Matrix &src, int row, int col, int channel = 1);
	/**
	@brief 返回随机抽取num次矩阵src的行或列组成的矩阵
	@param src 矩阵
	@param rc 抽取方式
	@param num 抽取次数
	*/
	const Matrix mRandSample(const Matrix &src, X_Y_Z rc, int num = 1);
	/**
	@brief 返回从low到top等分成的1*len的矩阵
	@param low 下界
	@param top 上界
	@param len 等分个数
	*/
	const Matrix linspace(int low, int top, int len);
	/**
	@brief 返回从low到top等分成的1*len的矩阵
	@param low 下界
	@param top 上界
	@param len 等分个数
	*/
	const Matrix linspace(double low, double top, int len);
	/**
	@brief 返回矩阵的伴随矩阵
	@param src 矩阵
	*/
	const Matrix adj(const Matrix &src);
	/**
	@brief 返回矩阵的逆矩阵
	@param src 矩阵
	*/
	const Matrix inv(const Matrix &src);
	/**
	@brief 返回矩阵的伪逆矩阵
	@param src 矩阵
	@param dire 伪逆矩阵的计算方式
	*/
	const Matrix pinv(const Matrix &src, direction dire = LEFT);
	/**
	@brief 返回矩阵的转置矩阵
	@param src 矩阵
	*/
	const Matrix tran(const Matrix &src);
	/**
	@brief 返回矩阵的绝对值矩阵
	@param src 矩阵
	*/
	const Matrix mAbs(const Matrix &src);
	/**
	@brief 返回angle度2*2的旋转矩阵
	@param angle 角度
	*/
	const Matrix Rotate(double angle);
	/**
	@brief 返回矩阵num次幂
	@param src 矩阵
	@param num 次幂
	*/
	const Matrix POW(const Matrix &src, int num);
	/**
	@brief 返回矩阵取反
	@param src 矩阵
	*/
	const Matrix mOpp(const Matrix &src);
	/**
	@brief 返回矩阵按行或列之和
	@param src 矩阵
	@param rc 求和的方向
	*/
	const Matrix mSum(const Matrix &src, X_Y_Z rc);
	/**
	@brief 返回矩阵按元素取指数
	@param src 矩阵
	*/
	const Matrix mExp(const Matrix &src);
	/**
	@brief 返回矩阵按元素取对数
	@param src 矩阵
	*/
	const Matrix mLog(const Matrix &src);
	/**
	@brief 返回矩阵按元素取开方
	@param src 矩阵
	*/
	const Matrix mSqrt(const Matrix &src);
	/**
	@brief 返回矩阵按元素取num次幂
	@param src 矩阵
	@param num 次幂
	*/
	const Matrix mPow(const Matrix &src, int num);
	/**
	@brief 返回矩阵val/src按元素除
	@param src 矩阵
	@param val 除数
	*/
	const Matrix Divi(const Matrix &src, double val, direction dire = RIGHT);
	/**
	@brief 返回矩阵除法
	@param a 被除矩阵
	@param b 除矩阵
	@param dire 除法方式
	*/
	const Matrix Divi(const Matrix &a, const Matrix &b, direction dire = RIGHT);
	/**
	@brief 返回矩阵按元素对乘
	@param a 矩阵
	@param b 矩阵
	*/
	const Matrix Mult(const Matrix &a, const Matrix &b);
	/**
	@brief 返回矩阵按元素取a和b之间的最大值
	@param a 比较值
	@param b 比较矩阵
	*/
	const Matrix mMax(double a, const Matrix &b);
	/**
	@brief 返回矩阵按元素取a和b之间的最大值
	@param a 比较矩阵
	@param b 比较矩阵
	*/
	const Matrix mMax(const Matrix &a, const Matrix &b);
	/**
	@brief 返回矩阵按元素取a和b之间的最小值
	@param a 比较值
	@param b 比较矩阵
	*/
	const Matrix mMin(double a, const Matrix &b);
	/**
	@brief 返回矩阵按元素取a和b之间的最小值
	@param a 比较矩阵
	@param b 比较矩阵
	*/
	const Matrix mMin(const Matrix &a, const Matrix &b);	
	/**
	@brief mCalSize 计算卷积所需扩张的边界
	返回矩阵大小
	@param src 被卷积矩阵
	@param kern 卷积核
	@param anchor 像素对应卷积核坐标
	anchor默认为Point(-1,-1), 像素对应卷积核中心
	@param strides 滑动步长
	@param top 向上扩充几行
	@param bottom 向下扩充几行
	@param left 向左扩充几列
	@param right 向右扩充几列
	*/
	Size3 mCalSize(const Matrix &src, const Matrix &kern, Point & anchor, Size strides, int &top, int &bottom, int &left, int &right);
	/**
	@brief mCalSize 计算卷积所需扩张的边界
	返回矩阵大小
	@param src 被卷积矩阵尺寸
	@param kern 卷积核尺寸
	@param anchor 像素对应卷积核坐标
	*/
	Size3 mCalSize(Size3 src, Size3 kern, Point & anchor, Size strides);
	/**
	@brief 返回按boundary分界填充的矩阵
	返回矩阵大小等于输入矩阵大小
	@param src 输入矩阵
	@param boundary 分界值
	@param lower 小于boundary用lower填充
	@param upper 大于boundary用upper填充
	@param boundary2upper 当矩阵元素等于boundary时
	为1归upper, 为-1归lower, 为0不处理
	*/
	const Matrix mThreshold(const Matrix &src, double boundary, double lower, double upper, int boundary2upper = 1);
	/**
	@brief 返回边界扩充的矩阵
	@param src 输入矩阵
	@param top 向上扩充几行
	@param bottom 向下扩充几行
	@param left 向左扩充几列
	@param right 向右扩充几列
	@param borderType 边界像素外推的插值方法
	@param value 常量插值的数值
	**/
	const Matrix copyMakeBorder(const Matrix &src, int top, int bottom, int left, int right, BorderTypes borderType = BORDER_CONSTANT, const int value = 0);
	/**
	@brief 返回矩阵2维卷积结果
	返回矩阵大小为(input.row/strides_x, input.col/strides_y, 1)
	@param input 输入矩阵
	@param kern 卷积核
	@param anchor 矩阵元素对应卷积核的位置
	以卷积核的左上角为(0,0)点, 默认(-1,-1)为中心
	@param strides 滑动步长 
	Size.hei为x轴,Size.wid为y轴
	@param is_copy_border 是否要扩展边界
	*/
	const Matrix Filter2D(const Mat & input, const Mat & kern, Point anchor = Point(-1, -1), const Size & strides = Size(1, 1), bool is_copy_border = true);
	/**
	最小二乘法
	@param x 自变量
	@param y 因变量
	*/
	const Matrix LeastSquare(const Matrix& x, const Matrix &y);
	/**
	@brief 命令行按矩阵输出
	@param row 行
	@param col 列
	*/
	template<typename T>
	void showMatrix(const T *, int row, int col); 
	void pause();
}

#endif //__MAT_H__
