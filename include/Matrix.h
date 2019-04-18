#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <vector>
#include "Vriable.h"

namespace nn {
	class MatCommaInitializer_;
	class Matrix
	{
	public:
		explicit Matrix();
		/**
		@brief 生成1*1*1的方阵,用value填充
		@param value
		*/
		template<class Type>
		Matrix(Type value)
		{
			init();
			*this = Matrix(1, 1, 1);
			matrix[0] = (double)value;
		}
		/**
		@brief 生成row*col*1的方阵,用0填充
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		Matrix(int row, int col);
		/**
		@brief 生成row*col*depth的方阵,用0填充
		@param row 矩阵行数
		@param col 矩阵列数
		@param depth 矩阵通道数
		*/
		Matrix(int row, int col, int depth);
		/**
		@brief 生成size*1的方阵,用0填充
		@param size_ 矩阵尺寸
		*/
		Matrix(Size size_);
		/**
		@brief 生成size的方阵,用0填充
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
		Matrix(Matrix a, Matrix b, X_Y_Z merge);
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
		Matrix(double *matrix, int n);
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
		Matrix(double *matrix, int row, int col, int channel = 1);
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
				mat(idx++) = (double)v;
			*this = mat;
		}
		~Matrix();
		/**
		@brief 返回矩阵指针
		*/
		double* mat_()const;
		/**
		@brief 检查维度
		*/
		void DimCheck()const;
		/**
		@brief 返回矩阵尺寸(row,col,channel)
		*/
		Size3 size3()const;
		/**
		@brief 返回矩阵行数
		*/
		int rows()const;
		/**
		@brief 返回矩阵列数
		*/
		int cols()const;
		/**
		@brief 返回矩阵通道数
		*/
		int channels()const;
		/**
		@brief 返回矩阵大小(row*col*1)
		*/
		int size()const;
		/**
		@brief 返回矩阵大小Size(row,col)
		*/
		Size mSize()const;
		/**
		@brief 返回矩阵大小(row*col*1)
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
		@brief 按索引返回矩阵元素
		@param index 索引
		*/
		double& at(int index)const;
		/**
		@brief 按索引返回矩阵元素
		@param index_x 行索引
		@param index_y 列索引
		*/
		double& at(int index_y, int index_x)const;
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
		double frist()const;
		/**
		@brief 返回矩阵与value相等的第一个元素索引
		@param value 元素
		*/
		int find(double value)const;
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
		bool contains(double value)const;
		/**
		@brief 返回矩阵与value相等的第一个元素
		@param value 元素
		*/
		double& findAt(double value)const;
		/**
		@brief 返回矩阵元素最大值
		*/
		double& findmax()const;
		/**
		@brief 返回矩阵元素最小值
		*/
		double& findmin()const;
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
		bool setSize(int row, int col);
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
		void setNum(double number, int index);
		/**
		@brief 修改矩阵对应索引元素
		@param number 元素
		@param index_y 行索引
		@param index_x 列索引
		*/
		void setNum(double number, int index_y, int index_x);
		/**
		@brief 重置矩阵
		@param mat 矩阵元素
		@param row 矩阵行数
		@param col 矩阵列数
		*/
		void setMat(double *mat, int row, int col);
		/**
		@brief 设置逆矩阵
		*/
		void setInv();
		/**
		@brief 设置矩阵的num次幂
		@param num 次幂
		*/
		void setPow(int num);
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
		@brief 返回取反矩阵
		*/
		const Matrix Opp()const;
		/**
		@brief 返回绝对值矩阵
		*/
		const Matrix Abs()const;
		/**
		@brief 返回按num次幂矩阵
		@param num 次幂
		*/
		const Matrix Pow(int num)const;
		/**
		@brief 返回按元素取指数矩阵
		*/
		const Matrix Exp()const;
		/**
		@brief 返回按元素取对数矩阵
		*/
		const Matrix Log()const;
		/**
		@brief 返回按元素取开方矩阵
		*/
		const Matrix Sqrt()const;
		/**
		@brief 返回伴随矩阵
		*/
		const Matrix Adj()const;
		/**
		@brief 返回转置矩阵
		*/
		const Matrix t()const;
		/**
		@brief 返回逆矩阵
		*/
		const Matrix Inv()const;
		/**
		@brief 返回逆序矩阵
		矩阵必须是向量
		*/
		const Matrix Reverse()const;
		const Matrix EigenvectorsMax(double offset = 1e-8)const;

		/**
		@brief 返回行列式
		*/
		double Det();
		/**
		@brief 返回num范数
		@param num 几范数
		*/
		double Norm(int num = 1)const;
		/**
		@brief 返回对应索引的余子式
		@param x 列索引
		@param y 行索引
		*/
		double Cof(int x, int y);
		double EigenvalueMax(double offset = 1e-8)const;
		/**
		@brief 返回随机抽取的矩阵元素
		*/
		double RandSample();
		/**
		@brief 返回矩阵元素和
		@param num 设置次幂
		@param _abs 是否取绝对值
		*/
		double Sum(int num = 1, bool _abs = false)const;
		/**
		@brief 重载运算符+
		对应元素相加
		*/
		const Matrix operator + (const double val)const;
		/**
		@brief 重载运算符+
		对应元素相加
		*/
		const Matrix operator + (const Matrix &a)const;
		/**
		@brief 重载运算符+=
		按元素相加
		*/
		void operator += (const double val);
		/**
		@brief 重载运算符+=
		按元素相加
		*/
		void operator += (const Matrix &a);
		/**
		@brief 友元重载运算符+
		按元素相加
		*/
		friend const Matrix operator + (const double value, const Matrix &mat);
		/**
		@brief 重载运算符-
		按元素取相反数
		*/
		const Matrix operator - (void)const;
		/**
		@brief 重载运算符-
		按元素相减
		*/
		const Matrix operator - (const double val)const;
		/**
		@brief 重载运算符-
		对应元素相减
		*/
		const Matrix operator - (const Matrix &a)const;
		/**
		@brief 重载运算符-=
		按元素相减
		*/
		void operator -= (const double val);
		/**
		@brief 重载运算符-=
		对应元素相减
		*/
		void operator -= (const Matrix &a);
		/**
		@brief 友元重载运算符-
		按元素相减
		*/
		friend const Matrix operator - (const double value, const Matrix &mat);
		/**
		@brief 重载运算符*
		按元素相乘
		*/
		const Matrix operator * (const double val)const;
		/**
		@brief 重载运算符*
		对应元素相乘
		*/
		const Matrix operator * (const Matrix &a)const;
		/**
		@brief 重载运算符*=
		按元素相乘
		*/
		void operator *= (const double val);
		/**
		@brief 重载运算符*=
		对应元素相乘
		*/
		void operator *= (const Matrix &a);
		/**
		@brief 友元重载运算符*
		按元素相乘
		*/
		friend const Matrix operator * (const double value, const Matrix &mat);
		/**
		@brief 重载运算符/
		按元素相除
		*/
		const Matrix operator / (const double val)const;
		/**
		@brief 重载运算符/
		矩阵乘法
		*/
		const Matrix operator / (const Matrix &a)const;
		/**
		@brief 重载运算符/=
		按元素相除
		*/
		void operator /= (const double val);
		/**
		@brief 重载运算符/=
		对应元素相除
		*/
		void operator /= (const Matrix &a);
		/**
		@brief 友元重载运算符/
		按元素相乘
		*/
		friend const Matrix operator / (const double value, const Matrix &mat);
		/**
		@brief 重载运算符=
		深拷贝
		*/
		void operator = (const Matrix &temp);
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
		double& operator () (const int index)const;
		/**
		@brief 返回对应索引元素
		@param row 行索引
		@param col 列索引
		*/
		double& operator () (const int row, const int col)const;
		/**
		@brief 返回对应索引元素
		@param row 行索引
		@param col 列索引
		@param depth 通道索引
		*/
		double& operator () (const int row, const int col, const int depth)const;
		/**
		@brief 返回矩阵对应索引的列或行
		@param index 索引
		@param rc 索引方式
		*/
		const Matrix operator () (const int index, X_Y_Z rc)const;
		/**
		@brief 返回矩阵对应通道索引
		@param channel 通道索引
		*/
		const Matrix operator [] (const int channel)const;
		friend std::ostream & operator << (std::ostream &out, const Matrix &ma);

	private:
		int row;
		int col;
		int depth;
		bool square;
		double *matrix;

		void init();
		void checkSquare();
		void checkindex(int index)const;
		void checkindex(int index_x, int index_y)const;
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
		@brief 生成row*col*channel的方阵,用0填充
		@param row 矩阵行数
		@param col 矩阵列数
		@param depth 矩阵通道数
		*/
		Mat_(int row, int col = 1, int channel = 1) : Mat(row, col, channel) {}
		/**
		@brief 生成size_[0]*size_[1]*size_[2]的方阵,用0填充
		会释放Vec<int> size_的内存
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
		double * matrix()const { return head; }
	private:
		int row;
		int col;
		int channel;
		double *it;
		double *head;
	};
	template<typename Tp_>
	inline MatCommaInitializer_ MatCommaInitializer_::operator , (Tp_ v)
	{
		if (this->it == this->head + row * col*channel) {
			fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			throw MatCommaInitializer_();
		}
		*this->it = double(v);
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