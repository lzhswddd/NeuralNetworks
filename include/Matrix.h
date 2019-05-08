#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>
#include "vriable.h"
#include "alignMalloc.h"

//#define DEEPCOPY_MAT
#ifndef DEEPCOPY_MAT
#define LIGHT_MAT
#endif // !DEEPCOPY_MAT

#ifdef _DEBUG
#define MAT_DEBUG
#else
#define MAT_RELEASE
#endif

namespace nn {
	enum MatType
	{
		NN_CHAR = 0,
		NN_UCHAR,
		NN_INT,
		NN_UINT,
		NN_FLAOT,
		NN_DOUBLE,
	};
	class MatCommaInitializer_;
	class Matrix
	{
	public:
		explicit Matrix();
		/**
		@brief 生成1*col*1的方阵
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		Matrix(int w);
		/**
		@brief 生成row*col*1的方阵
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		Matrix(int row, int col);
		/**
		@brief 生成row*col*depth的方阵
		@param row 矩阵行数
		@param col 矩阵列数
		@param depth 矩阵通道数
		*/
		Matrix(int row, int col, int depth);
		/**
		@brief 生成size*1的方阵
		@param size_ 矩阵尺寸
		*/
		Matrix(Size size_);
		/**
		@brief 生成size的方阵
		@param size_ 矩阵尺寸
		*/
		Matrix(Size3 size_);
		/**
		@brief 拷贝函数
		@param src 拷贝对象
		*/
		Matrix(const Matrix *src);
		/**
		拷贝函数
		@param src 拷贝对象
		*/
		Matrix(const Matrix &src);
		/**
		@brief 将矩阵a和b合并(COL为按列合并|ROW为按行合并)
		@param a 输入矩阵1
		@param b 输入矩阵2
		@param merge 合并方式
		*/
		Matrix(const Matrix &a, const Matrix &b, X_Y_Z merge);
		/**
		@brief 构造函数
		深拷贝m
		@param m 矩阵
		*/
		Matrix(MatCommaInitializer_ &m);
		/**
		@brief 生成n*n*1的方阵,元素为matrix
		@param matrix 矩阵元素
		@param n 矩阵大小
		*/
		Matrix(int *matrix, int n);
		/**
		@brief 生成n*n*1的方阵,元素为matrix
		@param matrix 矩阵元素
		@param n 矩阵大小
		*/
		Matrix(float *matrix, int n);
		/**
		@brief 生成row*col*1的方阵,元素为matrix
		@param matrix 矩阵元素
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		Matrix(int *matrix, int row, int col, int channel = 1);
		/**
		@brief 生成row*col*1的方阵,元素为matrix
		@param matrix 矩阵元素
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		Matrix(float *matrix, int row, int col, int channel = 1);
		Matrix(int w, float *data);
		Matrix(int w, int h, float *data);
		Matrix(int w, int h, int c, float *data);
		Matrix(int w, int h, int c, int c_offset,  float *data);
		template<class Type>
		Matrix(const std::vector<Type> &vec, X_Y_Z dirc)
		{
			init();
			Matrix mat;
			switch (dirc)
			{
			case nn::ROW:mat = zeros((int)vec.size(), 1, 1);
				break;
			case nn::COL:mat = zeros(1, (int)vec.size(), 1);
				break;
			case nn::CHANNEL:mat = zeros(1, 1, (int)vec.size());
				break;
			default:
				return;
			}
			int idx = 0;
			for (const Type&v : vec)
				mat(idx++) = (float)v;
			*this = mat;
		}
		~Matrix();
		void create(int w);
		void create(int h, int w);
		void create(int h, int w, int c);
		void create(Size size);
		void create(Size3 size);
		/**
		@brief 返回矩阵指针
		*/
		float* mat_()const;
		/**
		@brief 检查维度
		*/
		void DimCheck()const;
		/**
		@brief 返回矩阵尺寸(row,col,channel)
		*/
		Size3 size3()const;
		/**
		@brief 返回矩阵偏移
		*/
		int total()const;
		/**
		@brief 返回维度
		*/
		int dims()const;
		/**
		@brief 返回行数
		*/
		int rows()const;
		/**
		@brief 返回列数
		*/
		int cols()const;
		/**
		@brief 返回通道数
		*/
		int channels()const;
		/**
		@brief 返回矩阵大小(row*col*channel)
		*/
		size_t size()const;
		/**
		@brief 返回矩阵大小Size(row,col)
		*/
		Size mSize()const;
		/**
		@brief 保存矩阵
		@param file 保存文件名
		@param binary 选择文本还是二进制
		binary = false 选择文本
		binary = true 选择二进制
		*/
		void save(std::string file, bool binary=true)const;
		/**
		@brief 读取矩阵
		@param file 读取文件名
		只支持二进制读取
		*/
		void load(std::string file);
		/**
		@brief 返回矩阵大小(row*col*channel)
		*/
		int length()const;
		/**
		@brief 返回矩阵状态
		0为方阵
		-1为空矩阵
		-2为非方阵
		*/
		int isEnable()const;
		/**
		@brief 返回矩阵是否为空
		*/
		bool empty()const;
		/**
		@brief 返回矩阵是否为方阵
		*/
		bool Square()const;
		/**
		@brief 拷贝
		*/
		void copyTo(const Matrix& mat)const;
		/**
		@brief 释放内存
		*/
		void release();
		/**
		@brief 按索引返回矩阵元素
		@param index 索引
		*/
		float& at(int index)const;
		/**
		@brief 按索引返回矩阵元素
		@param index_x 行索引
		@param index_y 列索引
		*/
		float& at(int index_y, int index_x)const;
		/**
		@brief 将索引转换为对应矩阵列索引
		@param index 索引
		*/
		int toX(int index)const;
		/**
		@brief 将索引转换为对应矩阵行索引
		@param index 索引
		*/
		int toY(int index)const;

		/**
		@brief 矩阵第一个元素
		*/
		float frist()const;
		/**
		@brief 返回矩阵与value相等的第一个元素索引
		@param value 元素
		*/
		int find(float value)const;
		/**
		@brief 返回矩阵元素最大值的索引
		*/
		int maxAt()const;
		/**
		@brief 返回矩阵元素最小值的索引
		*/
		int minAt()const;
		/**
		@brief 返回矩阵是否包含value
		@param value 元素
		*/
		bool contains(float value)const;
		/**
		@brief 返回矩阵与value相等的第一个元素
		@param value 元素
		*/
		float& findAt(float value)const;
		/**
		@brief 返回矩阵元素最大值
		*/
		float& findmax()const;
		/**
		@brief 返回矩阵元素最小值
		*/
		float& findmin()const;
		/**
		@brief 将矩阵按索引区域拷贝元素到src矩阵中
		@param src 被拷贝矩阵
		@param Row_Start 截取行初始索引值
		@param Row_End 截取行结束索引值
		@param Col_Start 截取列初始索引值
		@param Col_End 截取列结束索引值
		*/
		void copy(Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End)const;
		/**
		@brief 将矩阵拷贝到src
		@param src 被拷贝矩阵
		*/
		void swap(Matrix &src)const;
		/**
		@brief 在矩阵最左边或最右边添加一列1
		@param dire 选择添加方式
		*/
		void addones(direction dire = LEFT);
		/**
		@brief mChannel 将src覆盖到第channel通道
		@param src 矩阵
		@param channel 通道数
		*/
		void mChannel(const Matrix &src, int channel);
		/**
		@brief mChannel 将src覆盖到第channel通道
		@param src 矩阵
		@param channel 通道数
		*/
		void mChannel(const Matrix &src, int row, int col);
		/**
		@brief 设置矩阵维度
		不允许改变矩阵长度
		*/
		void reshape(Size3 size);
		/**
		@brief 设置矩阵维度
		不允许改变矩阵长度
		*/
		void reshape(int row, int col, int channel);
		/**
		@brief 设置矩阵大小
		如果矩阵原大小不等于row*col*1则元素全部重置为0
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		bool setSize(int row, int col, int channel);
		/**
		@brief 拷贝矩阵src
		@param src 拷贝矩阵
		*/
		void setvalue(const Matrix &src);
		/**
		@brief 修改矩阵对应索引元素
		@param number 元素
		@param index 索引
		*/
		void setNum(float number, int index);
		/**
		@brief 修改矩阵对应索引元素
		@param number 元素
		@param index_y 行索引
		@param index_x 列索引
		*/
		void setNum(float number, int index_y, int index_x);
		/**
		@brief 重置矩阵
		@param mat 矩阵元素
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		void setMat(float *mat, int row, int col);
		/**
		@brief 设置逆矩阵
		*/
		void setInv();
		/**
		@brief 设置矩阵的num次幂
		@param num 次幂
		*/
		void setPow(float num);
		/**
		@brief 设置取反
		*/
		void setOpp();
		/**
		@brief 设置单位矩阵
		*/
		void setIden();
		/**
		@brief 设置伴随矩阵
		*/
		void setAdj();
		/**
		@brief 设置转置矩阵
		*/
		void setTran();

		/**
		@brief 命令行输出矩阵
		*/
		void show()const;

		/**
		@brief 返回c通道矩阵
		@param 通道索引
		*/
		Matrix Channel(int c);
		/**
		@brief 返回c通道矩阵
		@param 通道索引
		*/
		const Matrix Channel(int c)const;
		/**
		@brief 返回深拷贝矩阵
		*/
		const Matrix clone()const;
		/**
		@brief 返回取反矩阵
		*/
		const Matrix opp()const;
		/**
		@brief 返回绝对值矩阵
		*/
		const Matrix abs()const;
		/**
		@brief 返回按num次幂矩阵
		@param num 次幂
		*/
		const Matrix mPow(int num)const;
		/**
		@brief 返回按num次幂矩阵
		@param num 次幂
		*/
		const Matrix pow(float num)const;
		/**
		@brief 返回按元素取指数矩阵
		*/
		const Matrix exp()const;
		/**
		@brief 返回按元素取对数矩阵
		*/
		const Matrix log()const;
		/**
		@brief 返回按元素取开方矩阵
		*/
		const Matrix sqrt()const;
		/**
		@brief 返回伴随矩阵
		*/
		const Matrix adj()const;
		/**
		@brief 返回转置矩阵
		*/
		const Matrix t()const;
		/**
		@brief 返回逆矩阵
		*/
		const Matrix inv()const;
		/**
		@brief 返回逆序矩阵
		矩阵必须是向量
		*/
		const Matrix reverse()const;
		const Matrix EigenvectorsMax(float offset = 1e-8)const;
		/**
		@brief sigmoid函数
		详细情况见Function.h
		*/
		const Matrix sigmoid()const;
		/**
		@brief tanh函数
		详细情况见Function.h
		*/
		const Matrix tanh()const;
		/**
		@brief relu函数
		详细情况见Function.h
		*/
		const Matrix relu()const;
		/**
		@brief elu函数
		详细情况见Function.h
		*/
		const Matrix elu()const;
		/**
		@brief selu函数
		详细情况见Function.h
		*/
		const Matrix selu()const;
		/**
		@brief leaky_relu函数
		详细情况见Function.h
		*/
		const Matrix leaky_relu()const;
		/**
		@brief softmax函数
		详细情况见Function.h
		*/
		const Matrix softmax()const;
		/**
		@brief 返回行列式
		*/
		float Det();
		/**
		@brief 返回num范数
		@param num 几范数
		*/
		float Norm(int num = 1)const;
		/**
		@brief 返回对应索引的余子式
		@param x 列索引
		@param y 行索引
		*/
		float Cof(int x, int y);
		float EigenvalueMax(float offset = 1e-8)const;
		/**
		@brief 返回随机抽取的矩阵元素
		*/
		float RandSample();
		/**
		@brief 返回矩阵元素和
		@param num 设置次幂
		@param _abs 是否取绝对值
		*/
		float sum(int num = 1, bool _abs = false)const;
		/**
		@brief 返回平均值
		*/
		float mean()const;
		/**
		@brief 重载运算符+
		对应元素相加
		*/
		const Matrix operator + (const float val)const;
		/**
		@brief 重载运算符+
		对应元素相加
		*/
		const Matrix operator + (const Matrix &a)const;
		/**
		@brief 重载运算符+=
		按元素相加
		*/
		void operator += (const float val);
		/**
		@brief 重载运算符+=
		按元素相加
		*/
		void operator += (const Matrix &a);
		/**
		@brief 友元重载运算符+
		按元素相加
		*/
		friend const Matrix operator + (const float value, const Matrix &mat);
		/**
		@brief 重载运算符-
		按元素取相反数
		*/
		const Matrix operator - (void)const;
		/**
		@brief 重载运算符-
		按元素相减
		*/
		const Matrix operator - (const float val)const;
		/**
		@brief 重载运算符-
		对应元素相减
		*/
		const Matrix operator - (const Matrix &a)const;
		/**
		@brief 重载运算符-=
		按元素相减
		*/
		void operator -= (const float val);
		/**
		@brief 重载运算符-=
		对应元素相减
		*/
		void operator -= (const Matrix &a);
		/**
		@brief 友元重载运算符-
		按元素相减
		*/
		friend const Matrix operator - (const float value, const Matrix &mat);
		/**
		@brief 重载运算符*
		按元素相乘
		*/
		const Matrix operator * (const float val)const;
		/**
		@brief 重载运算符*
		对应元素相乘
		*/
		const Matrix operator * (const Matrix &a)const;
		/**
		@brief 重载运算符*=
		按元素相乘
		*/
		void operator *= (const float val);
		/**
		@brief 重载运算符*=
		对应元素相乘
		*/
		void operator *= (const Matrix &a);
		/**
		@brief 友元重载运算符*
		按元素相乘
		*/
		friend const Matrix operator * (const float value, const Matrix &mat);
		/**
		@brief 重载运算符/
		按元素相除
		*/
		const Matrix operator / (const float val)const;
		/**
		@brief 重载运算符/
		矩阵乘法
		*/
		const Matrix operator / (const Matrix &a)const;
		/**
		@brief 重载运算符/=
		按元素相除
		*/
		void operator /= (const float val);
		/**
		@brief 重载运算符/=
		对应元素相除
		*/
		void operator /= (const Matrix &a);
		/**
		@brief 友元重载运算符/
		按元素相乘
		*/
		friend const Matrix operator / (const float value, const Matrix &mat);
		/**
		@brief 重载运算符=
		深拷贝
		*/
		Matrix & operator = (const Matrix &temp);
		/**
		@brief 重载运算符==
		判断矩阵是否相等
		*/
		bool operator == (const Matrix &a)const;
		/**
		@brief 重载运算符!=
		判断矩阵是否不相等
		*/
		bool operator != (const Matrix &a)const;
		/**
		@brief 返回对应索引元素
		@param index 索引
		*/
		float& operator () (const int index)const;
		/**
		@brief 返回对应索引元素
		@param row 行索引
		@param col 列索引
		*/
		float& operator () (const int row, const int col)const;
		/**
		@brief 返回对应索引元素
		@param row 行索引
		@param col 列索引
		@param depth 通道索引
		*/
		float& operator () (const int row, const int col, const int depth)const;
		/**
		@brief 返回矩阵对应索引的列或行
		@param index 索引
		@param rc 索引方式
		*/
		const Matrix operator () (const int index, X_Y_Z rc)const;
		/**
		@brief 返回矩阵对应索引的列或行
		@param index 索引
		@param rc 索引方式
		*/
		const Matrix operator () (const int v1, const int v2, X_Y_Z rc)const;
		operator float *() {
			return matrix;
		}

		operator const float *() const {
			return matrix;
		}
		/**
		@brief 返回矩阵对应通道索引
		@param channel 通道索引
		*/
		const Matrix operator [] (const int channel)const;
		friend std::ostream & operator << (std::ostream &out, const Matrix &ma);

	private:
		int row;
		int col;
		int channel;
		int dim;
		bool square;
		int offset_c;
		float *matrix;
#ifdef LIGHT_MAT
		int *recount;
#endif
		void init();
#ifdef LIGHT_MAT
		void createCount();
#endif
		void checkSquare();
#ifdef MAT_DEBUG
		void checkindex(int index);
		void checkindex(int index_x, int index_y);
#endif // MAT_DEBUG
		void setsize(int h, int w, int c);
	};

	typedef Matrix Mat;
	/**
	@brief Mat_ 工具类
	继承Mat类，用于实现
	Mat mat = (Mat_(3, 3) <<
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1);
	*/
	class Mat_ : public Mat
	{
	public:
		explicit Mat_() {}
		/**
		@brief 生成row*col*channel的方阵
		@param row 矩阵行数
		@param col 矩阵列数
		@param depth 矩阵通道数
		*/
		Mat_(int row, int col = 1, int channel = 1) : Mat(row, col, channel) {}
		/**
		@brief 生成size_[0]*size_[1]*size_[2]的方阵
		@param size_ 矩阵尺寸
		*/
		Mat_(const Size3 &size_) : Mat(size_) {}
	};
	/**
	@brief MatCommaInitializer_ 工具类
	作为迭代器，用于实现
	Mat mat = (Mat_(3, 3) <<
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1);
	*/
	class MatCommaInitializer_
	{
	public:
		explicit MatCommaInitializer_() {}
		MatCommaInitializer_(const Mat_& m) {
			head = m.mat_();
			it = head;
			row = m.rows();
			col = m.cols();
			channel = m.channels();
		}
		template<typename Tp_>
		MatCommaInitializer_ operator , (Tp_ v);
		int rows()const { return row; }
		int cols()const { return col; }
		int channels()const { return channel; }
		float * matrix()const { return head; }
	private:
		int row;
		int col;
		int channel;
		float *it;
		float *head;
	};
	template<typename Tp_>
	inline MatCommaInitializer_ MatCommaInitializer_::operator , (Tp_ v)
	{
#ifdef MAT_DEBUG
		if (this->it == this->head + row * col*channel) {
			fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			throw std::exception(errinfo[ERR_INFO_MEMOUT]);
		}
#endif
		*this->it = float(v);
		++this->it;
		return *this;
	}

	template<typename Tp_>
	static MatCommaInitializer_ operator << (const Mat_& m, Tp_ val)
	{
		MatCommaInitializer_ commaInitializer(m);
		return (commaInitializer, val);
	}
}

#endif //  __MATRIX_H__