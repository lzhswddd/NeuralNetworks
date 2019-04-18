#ifndef __VARIABLE_H__
#define __VARIABLE_H__
#include <stdio.h>

namespace nn
{
	static const double pi = 3.1415926535897932384626433832795f;
	typedef unsigned char uchar;
	typedef unsigned int uint;	
	enum ActivationType {
		ACTI_SIGMOID,
		ACTI_TANH,
		ACTI_RELU,
		ACTI_ELU,
		ACTI_LReLU,
		ACTI_SELU,
		ACTI_SOFTMAX
	};
	/**
	CONV2D			卷积层
	MAX_POOL		最大值池化层
	AVERAGE_POOL	平均值池化层
	FULL_CONNECTION 全连接层
	ACTIVATION		激活层
	RESHAPE			重置维度层
	DROPOUT			随机使能层
	*/
	enum LayerType {
		CONV2D = 0,
		MAX_POOL,
		AVERAGE_POOL,
		FULL_CONNECTION,
		ACTIVATION,
		RESHAPE,
		DROPOUT
	};
	/**
	None				不提供优化功能
	GradientDescent		提供梯度下降法
	Momentum			提供动量梯度下降法
	NesterovMomentum	提供预测动量梯度下降法
	Adagrad				提供自适应学习率梯度下降法
	RMSProp				提供改良自适应学习率梯度下降法
	Adam				提供自适应学习率动量梯度下降法
	NesterovAdam		提供自适应学习率预测动量梯度下降法
	*/
	enum OptimizerMethod
	{
		None = 0,		//!< 不提供优化功能
		GradientDescent,//!< 提供梯度下降法
		Momentum,		//!< 提供动量梯度下降法
		NesterovMomentum,//!< 提供预测动量梯度下降法
		Adagrad,		//!< 提供自适应学习率梯度下降法
		RMSProp,		//!< 提供改良自适应学习率梯度下降法
		Adam,			//!< 提供自适应学习率动量梯度下降法
		NesterovAdam	//!< 提供自适应学习率预测动量梯度下降法
	};
	enum BorderTypes {
		BORDER_CONSTANT = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
		BORDER_REPLICATE = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
		BORDER_REFLECT = 2, //!< `fedcba|abcdefgh|hgfedcb`
		BORDER_WRAP = 3, //!< `cdefgh|abcdefgh|abcdefg`
		BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
		BORDER_TRANSPARENT = 5, //!< `uvwxyz|abcdefgh|ijklmno`
		BORDER_ISOLATED = 16 //!< do not look outside of ROI
	};
	enum MatErrorInfo {
		ERR_INFO_EMPTY = 0,
		ERR_INFO_SQUARE,
		ERR_INFO_ADJ,
		ERR_INFO_INV,
		ERR_INFO_POW,
		ERR_INFO_IND,
		ERR_INFO_CON,
		ERR_INFO_EIGEN,
		ERR_INFO_LEN,
		ERR_INFO_MEMOUT,
		ERR_INFO_UNLESS,
		ERR_INFO_SIZE,
		ERR_INFO_MULT,
		ERR_INFO_NORM,
		ERR_INFO_VALUE,
		ERR_INFO_PINV,
		ERR_INFO_DET,
		ERR_INFO_DIM,
	};
	static const char *errinfo[] = {
		"error 0: 矩阵为空!",
		"error 1: 矩阵不是方阵！",
		"error 2: 矩阵不是方阵，不能设置伴随矩阵！",
		"error 3: 矩阵不是方阵，不能设置逆矩阵！",
		"error 4: 矩阵不是方阵，不能进行次幂运算！",
		"error 5: 矩阵不是方阵，不能设置为单位矩阵！",
		"error 6: 矩阵不收敛!",
		"error 7: 矩阵没有实数特征值!",
		"error 8: 矩阵维度为0！",
		"error 9: 矩阵索引出界！",
		"error 10: 矩阵索引无效！",
		"error 11: 两个矩阵维度不一致！",
		"error 12: 两个矩阵维度不满足乘法条件！",
		"error 13: 矩阵维度不为1，不是向量！",
		"error 14: 参数违法！",
		"error 15: 计算逆矩阵失败！"
		"error 16: 行列式为0！",
		"error 17: 不支持三维操作！"
	};
	enum X_Y_Z {
		ROW = 0,
		COL,
		CHANNEL
	};
	enum direction {
		LEFT = 0,
		RIGHT
	};
	class Rect
	{
	public:
		Rect() : x(0), y(0), width(0), height(0) {}
		Rect(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}
		int x;
		int y;
		int width;
		int height;
	};
	class Size
	{
	public:
		Size() :hei(0), wid(0) {}
		Size(int height, int width) :hei(height), wid(width) {}
		~Size() {}
		int hei;
		int wid;
	};
	class Size3
	{
	public:
		explicit Size3() : x(0), y(0), z(0) {}
		Size3(int x, int y, int z = 1) : x(x), y(y), z(z) {}
		int x;
		int y;
		int z;
	};
	template<class Tp_>
	class Point2
	{
	public:
		Point2() :x(), y() {}
		Point2(Tp_ x, Tp_ y) :x(x), y(y) {}
		~Point2() {}
		bool operator == (const Point2<Tp_> &P)const
		{
			return (x == P.x) && (y == P.y);
		}
		bool operator != (const Point2<Tp_> &P)const
		{
			return !((*this) == P);
		}
		Tp_ x;
		Tp_ y;		
	};
	template<typename Tp_, typename T2>
	const Point2<Tp_> operator + (const Point2<Tp_> &P, const T2& v)
	{
		return Point2<Tp_>(P.x + v, P.y + v);
	}
	template<typename Tp_>
	const Point2<Tp_> operator + (const Point2<Tp_> &P1, const Point2<Tp_>& P2)
	{
		return Point2<Tp_>(P1.x + P2.x, P1.y + P2.y);
	}
	template<typename Tp_, typename T2>
	const Point2<Tp_> operator - (const Point2<Tp_> &P, const T2& v)
	{
		return Point2<Tp_>(P.x - v, P.y - v);
	}
	template<typename Tp_>
	const Point2<Tp_> operator - (const Point2<Tp_> &P1, const Point2<Tp_>& P2)
	{
		return Point2<Tp_>(P1.x - P2.x, P1.y - P2.y);
	}
	template<typename Tp_, typename T2>
	const Point2<Tp_> operator * (const Point2<Tp_> &P, const T2& v)
	{
		return Point2<Tp_>(P.x * v, P.y * v);
	}
	template<typename Tp_>
	const Tp_ operator * (const Point2<Tp_> &P1, const Point2<Tp_>& P2)
	{
		return P1.x * P2.x + P1.y * P2.y;
	}
	typedef Point2<char> Point2c;
	typedef Point2<uchar> Point2Uc;
	typedef Point2<int> Point2i;
	typedef Point2<uint> Point2Ui;
	typedef Point2<float> Point2f;
	typedef Point2<double> Point2d;
	typedef Point2i Point;

	template<class Tp_>
	class Point3
	{
	public:
		Point3() :x(), y(), z() {}
		Point3(Tp_ x, Tp_ y, Tp_ z) :x(x), y(y), z(z) {}
		~Point3() {}
		bool operator == (Point2<Tp_> &P)const
		{
			return (x == P.x) && (y == P.y) && (z == P.z);
		}
		bool operator != (Point2<Tp_> &P)const
		{
			return !((*this) == P);
		}
		Tp_ x;
		Tp_ y;
		Tp_ z;
	};
	template<typename Tp_, typename T2>
	const Point3<Tp_> operator + (const Point3<Tp_> &P, const T2& v)
	{
		return Point3<Tp_>(P.x + v, P.y + v, P.z + v);
	}
	template<typename Tp_>
	const Point3<Tp_> operator + (const Point3<Tp_> &P1, const Point3<Tp_>& P2)
	{
		return Point3<Tp_>(P1.x + P2.x, P1.y + P2.y, P1.z + P2.z);
	}
	template<typename Tp_, typename T2>
	const Point3<Tp_> operator - (const Point3<Tp_> &P, const T2& v)
	{
		return Point3<Tp_>(P.x - v, P.y - v, P.z + v);
	}
	template<typename Tp_>
	const Point3<Tp_> operator - (const Point3<Tp_> &P1, const Point3<Tp_>& P2)
	{
		return Point2<Tp_>(P1.x - P2.x, P1.y - P2.y, P1.z - P2.z);
	}
	template<typename Tp_, typename T2>
	const Point3<Tp_> operator * (const Point3<Tp_> &P, const T2& v)
	{
		return Point3<Tp_>(P.x * v, P.y * v, P.z * v);
	}
	template<typename Tp_>
	const Tp_ operator * (const Point3<Tp_> &P1, const Point3<Tp_>& P2)
	{
		return P1.x * P2.x + P1.y * P2.y + P1.z * P2.z;
	}
	typedef Point3<char> Point3c;
	typedef Point3<uchar> Point3Uc;
	typedef Point3<int> Point3i;
	typedef Point3<uint> Point3Ui;
	typedef Point3<float> Point3f;
	typedef Point3<double> Point3d;
	class Color;
	template<class Type>
	class Vec
	{
	public:
		explicit Vec()
			: data(new Type[3]), row(3), col(1), channel(1)
		{
			isCreate = true;
			memset(data, 0, sizeof(Type)*3);
		}
		Vec(Type *data, int size = 3) : row(size), col(1), channel(1) {
			this->data = data;
		}
		Vec(Type d1, Type d2, Type d3) : data(new Type[3]), row(3), col(1), channel(1) {
			isCreate = true;
			data[0] = d1;
			data[1] = d2;
			data[2] = d3;
		}
		~Vec()
		{
			if (isCreate && data != nullptr) {
				release();
			}
		}
		void release()
		{
			delete[] data;
			data = nullptr;
			row = 0;
			col = 0;
			channel = 0;
		}
		Type& operator [](const int index)const {
			if (index < 0 || index >= row*col*channel) {
				fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			}
			return data[index];
		}
		Type& operator ()(const int x, const int y)const {
			if (x < 0 || x >= row || y < 0 || y >= col) {
				fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			}
			return data[(x*col + y)*channel];
		}
		Type& operator ()(const int x, const int y, const int z)const {
			if (x < 0 || x >= row || y < 0 || y >= col|| z < 0 || z >= channel) {
				fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			}
			return data[(x*col + y)*channel + z];
		}

		void operator = (const Vec<Type> &vec)
		{
			row = vec.row;
			col = vec.col;
			channel = vec.channel;
			memcpy(data, vec.data, sizeof(Type)*row*col*channel);
		}

		void operator = (Color &color)
		{
			if (row*col*channel == 1) {
				data[0] = color.r;
			}
			else if (row*col*channel == 3) {
				data[0] = color.r;
				data[1] = color.g;
				data[2] = color.b;
			}
		}
		int row;
		int col;
		int channel;
		Type* data;
	private:
		bool isCreate = false;
	};
}

#endif //__VARIABLE_H__
