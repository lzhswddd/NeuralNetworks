#ifndef __VARIABLE_H__
#define __VARIABLE_H__
#include <stdio.h>
#include <iostream>

namespace nn
{
	static const float pi = 3.1415926535897932384626433832795f;
	typedef unsigned char uchar;
	typedef unsigned int uint;		
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
		"error 0: 矩阵为空!\0",
		"error 1: 矩阵不是方阵!\0",
		"error 2: 矩阵不是方阵，不能设置伴随矩阵!\0",
		"error 3: 矩阵不是方阵，不能设置逆矩阵!\0",
		"error 4: 矩阵不是方阵，不能进行次幂运算!\0",
		"error 5: 矩阵不是方阵，不能设置为单位矩阵!\0",
		"error 6: 矩阵不收敛!\0",
		"error 7: 矩阵没有实数特征值!\0",
		"error 8: 矩阵维度为0!\0",
		"error 9: 矩阵索引出界!\0",
		"error 10: 矩阵索引无效!\0",
		"error 11: 两个矩阵维度不一致!\0",
		"error 12: 两个矩阵维度不满足乘法条件!\0",
		"error 13: 矩阵维度不为1，不是向量!\0",
		"error 14: 参数违法!\0",
		"error 15: 计算逆矩阵失败!\0"
		"error 16: 行列式为0!\0",
		"error 17: 不支持三维操作!\0"
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
	/**
	EqualIntervalSampling 等间隔采样
	LocalMean 局部均值
	*/
	enum ReductionMothed
	{
		EqualIntervalSampling = 0,
		LocalMean
	};
	/**
	EqualIntervalSampling 等间隔采样
	LocalMean 局部均值
	*/
	enum RotateAngle
	{
		ROTATE_90_ANGLE = 0,
		ROTATE_180_ANGLE,
		ROTATE_270_ANGLE
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
		int area()const {
			return width * height;
		}
		friend std::ostream & operator << (std::ostream &out, const Rect &t)
		{
			out << "Rect(" << t.x << "," << t.y << ","<< t.width << "," << t.height << ")";
			return out;
		}
	};
	class Size
	{
	public:
		Size() :h(0), w(0) {}
		Size(int height, int width) :h(height), w(width) {}
		~Size() {}
		int h;
		int w; 
		int area()const {
			return h * w;
		}
		friend std::ostream & operator << (std::ostream &out, const Size &t)
		{
			out << "Size(" << t.h << "," << t.w << ")";
			return out;
		}
	};
	class Size3
	{
	public:
		explicit Size3() : h(0), w(0), c(0) {}
		Size3(int x, int y, int z = 1) : h(x), w(y), c(z) {}
		int h;
		int w;
		int c;
		int area()const {
			return h * w * c;
		}
		friend std::ostream & operator << (std::ostream &out, const Size3 &t)
		{
			out << "Size(" << t.h << "," << t.w << "," << t.c << ")";
			return out;
		}
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
		friend std::ostream & operator << (std::ostream &out, const Point2<Tp_> &t)
		{
			out << "Point(" << t.x << "," << t.y << ")";
			return out;
		}
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
	typedef Point2<float> Point2d;
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
		friend std::ostream & operator << (std::ostream &out, const Point3<Tp_> &t)
		{
			out << "Point(" << t.x << "," << t.y << "," << t.z << ")";
			return out;
		}
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
	typedef Point3<float> Point3d;
	class Color
	{
	public:
		Color(uchar v) : r(v), g(v), b(v) {}
		Color(uchar r, uchar g, uchar b) : r(r), g(g), b(b) {}
		uchar r;
		uchar g;
		uchar b;
	};
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
