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
		@brief ����1*1*1�ķ���,��value���
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
		@brief ����row*col*1�ķ���,��0���
		@param row ��������
		@param col ��������
		*/
		Matrix(int row, int col);
		/**
		@brief ����row*col*depth�ķ���,��0���
		@param row ��������
		@param col ��������
		@param depth ����ͨ����
		*/
		Matrix(int row, int col, int depth);
		/**
		@brief ����size*1�ķ���,��0���
		@param size_ ����ߴ�
		*/
		Matrix(Size size_);
		/**
		@brief ����size�ķ���,��0���
		@param size_ ����ߴ�
		*/
		Matrix(Size3 size_);
		/**
		@brief ��������
		@param src ��������
		*/
		Matrix(const Matrix *src);
		/**
		��������
		@param src ��������
		*/
		Matrix(const Matrix &src);
		/**
		@brief ������a��b�ϲ�(COLΪ���кϲ�|ROWΪ���кϲ�)
		@param a �������1
		@param b �������2
		@param merge �ϲ���ʽ
		*/
		Matrix(Matrix a, Matrix b, X_Y_Z merge);
		/**
		@brief ���캯��
		���m
		@param m ����
		*/
		Matrix(MatCommaInitializer_ &m);
		/**
		@brief ����n*n*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param n �����С
		*/
		Matrix(int *matrix, int n);
		/**
		@brief ����n*n*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param n �����С
		*/
		Matrix(double *matrix, int n);
		/**
		@brief ����row*col*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param row ��������
		@param col ��������
		*/
		Matrix(int *matrix, int row, int col, int channel = 1);
		/**
		@brief ����row*col*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param row ��������
		@param col ��������
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
		@brief ���ؾ���ָ��
		*/
		double* mat_()const;
		/**
		@brief ���ά��
		*/
		void DimCheck()const;
		/**
		@brief ���ؾ���ߴ�(row,col,channel)
		*/
		Size3 size3()const;
		/**
		@brief ���ؾ�������
		*/
		int rows()const;
		/**
		@brief ���ؾ�������
		*/
		int cols()const;
		/**
		@brief ���ؾ���ͨ����
		*/
		int channels()const;
		/**
		@brief ���ؾ����С(row*col*1)
		*/
		int size()const;
		/**
		@brief ���ؾ����СSize(row,col)
		*/
		Size mSize()const;
		/**
		@brief ���ؾ����С(row*col*1)
		*/
		int length()const;
		/**
		@brief ���ؾ���״̬
		0Ϊ����
		-1Ϊ�վ���
		-2Ϊ�Ƿ���
		*/
		int isEnable()const;
		/**
		@brief ���ؾ����Ƿ�Ϊ��
		*/
		bool empty()const;
		/**
		@brief ���ؾ����Ƿ�Ϊ����
		*/
		bool Square()const;
		/**
		@brief ���������ؾ���Ԫ��
		@param index ����
		*/
		double& at(int index)const;
		/**
		@brief ���������ؾ���Ԫ��
		@param index_x ������
		@param index_y ������
		*/
		double& at(int index_y, int index_x)const;
		/**
		@brief ������ת��Ϊ��Ӧ����������
		@param index ����
		*/
		int toX(int index)const;
		/**
		@brief ������ת��Ϊ��Ӧ����������
		@param index ����
		*/
		int toY(int index)const;

		/**
		@brief �����һ��Ԫ��
		*/
		double frist()const;
		/**
		@brief ���ؾ�����value��ȵĵ�һ��Ԫ������
		@param value Ԫ��
		*/
		int find(double value)const;
		/**
		@brief ���ؾ���Ԫ�����ֵ������
		*/
		int maxAt()const;
		/**
		@brief ���ؾ���Ԫ����Сֵ������
		*/
		int minAt()const;
		/**
		@brief ���ؾ����Ƿ����value
		@param value Ԫ��
		*/
		bool contains(double value)const;
		/**
		@brief ���ؾ�����value��ȵĵ�һ��Ԫ��
		@param value Ԫ��
		*/
		double& findAt(double value)const;
		/**
		@brief ���ؾ���Ԫ�����ֵ
		*/
		double& findmax()const;
		/**
		@brief ���ؾ���Ԫ����Сֵ
		*/
		double& findmin()const;
		/**
		@brief �������������򿽱�Ԫ�ص�src������
		@param src ����������
		@param Row_Start ��ȡ�г�ʼ����ֵ
		@param Row_End ��ȡ�н�������ֵ
		@param Col_Start ��ȡ�г�ʼ����ֵ
		@param Col_End ��ȡ�н�������ֵ
		*/
		void copy(Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End)const;
		/**
		@brief �����󿽱���src
		@param src ����������
		*/
		void swap(Matrix &src)const;
		/**
		@brief �ھ�������߻����ұ����һ��1
		@param dire ѡ����ӷ�ʽ
		*/
		void addones(direction dire = LEFT);
		/**
		@brief mChannel ��src���ǵ���channelͨ��
		@param src ����
		@param channel ͨ����
		*/
		void mChannel(const Matrix &src, int channel);
		/**
		@brief ���þ���ά��
		������ı���󳤶�
		*/
		void reshape(Size3 size);
		/**
		@brief ���þ���ά��
		������ı���󳤶�
		*/
		void reshape(int row, int col, int channel);
		/**
		@brief ���þ����С
		�������ԭ��С������row*col*1��Ԫ��ȫ������Ϊ0
		@param row ��������
		@param col ��������
		*/
		bool setSize(int row, int col);
		/**
		@brief ��������src
		@param src ��������
		*/
		void setvalue(const Matrix &src);
		/**
		@brief �޸ľ����Ӧ����Ԫ��
		@param number Ԫ��
		@param index ����
		*/
		void setNum(double number, int index);
		/**
		@brief �޸ľ����Ӧ����Ԫ��
		@param number Ԫ��
		@param index_y ������
		@param index_x ������
		*/
		void setNum(double number, int index_y, int index_x);
		/**
		@brief ���þ���
		@param mat ����Ԫ��
		@param row ��������
		@param col ��������
		*/
		void setMat(double *mat, int row, int col);
		/**
		@brief ���������
		*/
		void setInv();
		/**
		@brief ���þ����num����
		@param num ����
		*/
		void setPow(int num);
		/**
		@brief ����ȡ��
		*/
		void setOpp();
		/**
		@brief ���õ�λ����
		*/
		void setIden();
		/**
		@brief ���ð������
		*/
		void setAdj();
		/**
		@brief ����ת�þ���
		*/
		void setTran();

		/**
		@brief �������������
		*/
		void show()const;

		/**
		@brief ����ȡ������
		*/
		const Matrix Opp()const;
		/**
		@brief ���ؾ���ֵ����
		*/
		const Matrix Abs()const;
		/**
		@brief ���ذ�num���ݾ���
		@param num ����
		*/
		const Matrix Pow(int num)const;
		/**
		@brief ���ذ�Ԫ��ȡָ������
		*/
		const Matrix Exp()const;
		/**
		@brief ���ذ�Ԫ��ȡ��������
		*/
		const Matrix Log()const;
		/**
		@brief ���ذ�Ԫ��ȡ��������
		*/
		const Matrix Sqrt()const;
		/**
		@brief ���ذ������
		*/
		const Matrix Adj()const;
		/**
		@brief ����ת�þ���
		*/
		const Matrix t()const;
		/**
		@brief ���������
		*/
		const Matrix Inv()const;
		/**
		@brief �����������
		�������������
		*/
		const Matrix Reverse()const;
		const Matrix EigenvectorsMax(double offset = 1e-8)const;

		/**
		@brief ��������ʽ
		*/
		double Det();
		/**
		@brief ����num����
		@param num ������
		*/
		double Norm(int num = 1)const;
		/**
		@brief ���ض�Ӧ����������ʽ
		@param x ������
		@param y ������
		*/
		double Cof(int x, int y);
		double EigenvalueMax(double offset = 1e-8)const;
		/**
		@brief ���������ȡ�ľ���Ԫ��
		*/
		double RandSample();
		/**
		@brief ���ؾ���Ԫ�غ�
		@param num ���ô���
		@param _abs �Ƿ�ȡ����ֵ
		*/
		double Sum(int num = 1, bool _abs = false)const;
		/**
		@brief ���������+
		��ӦԪ�����
		*/
		const Matrix operator + (const double val)const;
		/**
		@brief ���������+
		��ӦԪ�����
		*/
		const Matrix operator + (const Matrix &a)const;
		/**
		@brief ���������+=
		��Ԫ�����
		*/
		void operator += (const double val);
		/**
		@brief ���������+=
		��Ԫ�����
		*/
		void operator += (const Matrix &a);
		/**
		@brief ��Ԫ���������+
		��Ԫ�����
		*/
		friend const Matrix operator + (const double value, const Matrix &mat);
		/**
		@brief ���������-
		��Ԫ��ȡ�෴��
		*/
		const Matrix operator - (void)const;
		/**
		@brief ���������-
		��Ԫ�����
		*/
		const Matrix operator - (const double val)const;
		/**
		@brief ���������-
		��ӦԪ�����
		*/
		const Matrix operator - (const Matrix &a)const;
		/**
		@brief ���������-=
		��Ԫ�����
		*/
		void operator -= (const double val);
		/**
		@brief ���������-=
		��ӦԪ�����
		*/
		void operator -= (const Matrix &a);
		/**
		@brief ��Ԫ���������-
		��Ԫ�����
		*/
		friend const Matrix operator - (const double value, const Matrix &mat);
		/**
		@brief ���������*
		��Ԫ�����
		*/
		const Matrix operator * (const double val)const;
		/**
		@brief ���������*
		��ӦԪ�����
		*/
		const Matrix operator * (const Matrix &a)const;
		/**
		@brief ���������*=
		��Ԫ�����
		*/
		void operator *= (const double val);
		/**
		@brief ���������*=
		��ӦԪ�����
		*/
		void operator *= (const Matrix &a);
		/**
		@brief ��Ԫ���������*
		��Ԫ�����
		*/
		friend const Matrix operator * (const double value, const Matrix &mat);
		/**
		@brief ���������/
		��Ԫ�����
		*/
		const Matrix operator / (const double val)const;
		/**
		@brief ���������/
		����˷�
		*/
		const Matrix operator / (const Matrix &a)const;
		/**
		@brief ���������/=
		��Ԫ�����
		*/
		void operator /= (const double val);
		/**
		@brief ���������/=
		��ӦԪ�����
		*/
		void operator /= (const Matrix &a);
		/**
		@brief ��Ԫ���������/
		��Ԫ�����
		*/
		friend const Matrix operator / (const double value, const Matrix &mat);
		/**
		@brief ���������=
		���
		*/
		void operator = (const Matrix &temp);
		/**
		@brief ���������==
		�жϾ����Ƿ����
		*/
		bool operator == (const Matrix &a)const;
		/**
		@brief ���������!=
		�жϾ����Ƿ����
		*/
		bool operator != (const Matrix &a)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param index ����
		*/
		double& operator () (const int index)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param row ������
		@param col ������
		*/
		double& operator () (const int row, const int col)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param row ������
		@param col ������
		@param depth ͨ������
		*/
		double& operator () (const int row, const int col, const int depth)const;
		/**
		@brief ���ؾ����Ӧ�������л���
		@param index ����
		@param rc ������ʽ
		*/
		const Matrix operator () (const int index, X_Y_Z rc)const;
		/**
		@brief ���ؾ����Ӧͨ������
		@param channel ͨ������
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
	@brief Mat_ ������
	�̳�Mat�࣬����ʵ��
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
		@brief ����row*col*channel�ķ���,��0���
		@param row ��������
		@param col ��������
		@param depth ����ͨ����
		*/
		Mat_(int row, int col = 1, int channel = 1) : Mat(row, col, channel) {}
		/**
		@brief ����size_[0]*size_[1]*size_[2]�ķ���,��0���
		���ͷ�Vec<int> size_���ڴ�
		@param size_ ����ߴ�
		*/
		Mat_(const Size3 &size_) : Mat(size_) {}
	};
	/**
	@brief MatCommaInitializer_ ������
	��Ϊ������������ʵ��
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