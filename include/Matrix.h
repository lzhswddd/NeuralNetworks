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
		@brief ����1*col*1�ķ���
		@param row ��������
		@param col ��������
		*/
		Matrix(int w);
		/**
		@brief ����row*col*1�ķ���
		@param row ��������
		@param col ��������
		*/
		Matrix(int row, int col);
		/**
		@brief ����row*col*depth�ķ���
		@param row ��������
		@param col ��������
		@param depth ����ͨ����
		*/
		Matrix(int row, int col, int depth);
		/**
		@brief ����size*1�ķ���
		@param size_ ����ߴ�
		*/
		Matrix(Size size_);
		/**
		@brief ����size�ķ���
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
		Matrix(const Matrix &a, const Matrix &b, X_Y_Z merge);
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
		Matrix(float *matrix, int n);
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
		@brief ���ؾ���ָ��
		*/
		float* mat_()const;
		/**
		@brief ���ά��
		*/
		void DimCheck()const;
		/**
		@brief ���ؾ���ߴ�(row,col,channel)
		*/
		Size3 size3()const;
		/**
		@brief ���ؾ���ƫ��
		*/
		int total()const;
		/**
		@brief ����ά��
		*/
		int dims()const;
		/**
		@brief ��������
		*/
		int rows()const;
		/**
		@brief ��������
		*/
		int cols()const;
		/**
		@brief ����ͨ����
		*/
		int channels()const;
		/**
		@brief ���ؾ����С(row*col*channel)
		*/
		size_t size()const;
		/**
		@brief ���ؾ����СSize(row,col)
		*/
		Size mSize()const;
		/**
		@brief �������
		@param file �����ļ���
		@param binary ѡ���ı����Ƕ�����
		binary = false ѡ���ı�
		binary = true ѡ�������
		*/
		void save(std::string file, bool binary=true)const;
		/**
		@brief ��ȡ����
		@param file ��ȡ�ļ���
		ֻ֧�ֶ����ƶ�ȡ
		*/
		void load(std::string file);
		/**
		@brief ���ؾ����С(row*col*channel)
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
		@brief ����
		*/
		void copyTo(const Matrix& mat)const;
		/**
		@brief �ͷ��ڴ�
		*/
		void release();
		/**
		@brief ���������ؾ���Ԫ��
		@param index ����
		*/
		float& at(int index)const;
		/**
		@brief ���������ؾ���Ԫ��
		@param index_x ������
		@param index_y ������
		*/
		float& at(int index_y, int index_x)const;
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
		float frist()const;
		/**
		@brief ���ؾ�����value��ȵĵ�һ��Ԫ������
		@param value Ԫ��
		*/
		int find(float value)const;
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
		bool contains(float value)const;
		/**
		@brief ���ؾ�����value��ȵĵ�һ��Ԫ��
		@param value Ԫ��
		*/
		float& findAt(float value)const;
		/**
		@brief ���ؾ���Ԫ�����ֵ
		*/
		float& findmax()const;
		/**
		@brief ���ؾ���Ԫ����Сֵ
		*/
		float& findmin()const;
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
		@brief mChannel ��src���ǵ���channelͨ��
		@param src ����
		@param channel ͨ����
		*/
		void mChannel(const Matrix &src, int row, int col);
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
		bool setSize(int row, int col, int channel);
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
		void setNum(float number, int index);
		/**
		@brief �޸ľ����Ӧ����Ԫ��
		@param number Ԫ��
		@param index_y ������
		@param index_x ������
		*/
		void setNum(float number, int index_y, int index_x);
		/**
		@brief ���þ���
		@param mat ����Ԫ��
		@param row ��������
		@param col ��������
		*/
		void setMat(float *mat, int row, int col);
		/**
		@brief ���������
		*/
		void setInv();
		/**
		@brief ���þ����num����
		@param num ����
		*/
		void setPow(float num);
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
		@brief ����cͨ������
		@param ͨ������
		*/
		Matrix Channel(int c);
		/**
		@brief ����cͨ������
		@param ͨ������
		*/
		const Matrix Channel(int c)const;
		/**
		@brief �����������
		*/
		const Matrix clone()const;
		/**
		@brief ����ȡ������
		*/
		const Matrix opp()const;
		/**
		@brief ���ؾ���ֵ����
		*/
		const Matrix abs()const;
		/**
		@brief ���ذ�num���ݾ���
		@param num ����
		*/
		const Matrix mPow(int num)const;
		/**
		@brief ���ذ�num���ݾ���
		@param num ����
		*/
		const Matrix pow(float num)const;
		/**
		@brief ���ذ�Ԫ��ȡָ������
		*/
		const Matrix exp()const;
		/**
		@brief ���ذ�Ԫ��ȡ��������
		*/
		const Matrix log()const;
		/**
		@brief ���ذ�Ԫ��ȡ��������
		*/
		const Matrix sqrt()const;
		/**
		@brief ���ذ������
		*/
		const Matrix adj()const;
		/**
		@brief ����ת�þ���
		*/
		const Matrix t()const;
		/**
		@brief ���������
		*/
		const Matrix inv()const;
		/**
		@brief �����������
		�������������
		*/
		const Matrix reverse()const;
		const Matrix EigenvectorsMax(float offset = 1e-8)const;
		/**
		@brief sigmoid����
		��ϸ�����Function.h
		*/
		const Matrix sigmoid()const;
		/**
		@brief tanh����
		��ϸ�����Function.h
		*/
		const Matrix tanh()const;
		/**
		@brief relu����
		��ϸ�����Function.h
		*/
		const Matrix relu()const;
		/**
		@brief elu����
		��ϸ�����Function.h
		*/
		const Matrix elu()const;
		/**
		@brief selu����
		��ϸ�����Function.h
		*/
		const Matrix selu()const;
		/**
		@brief leaky_relu����
		��ϸ�����Function.h
		*/
		const Matrix leaky_relu()const;
		/**
		@brief softmax����
		��ϸ�����Function.h
		*/
		const Matrix softmax()const;
		/**
		@brief ��������ʽ
		*/
		float Det();
		/**
		@brief ����num����
		@param num ������
		*/
		float Norm(int num = 1)const;
		/**
		@brief ���ض�Ӧ����������ʽ
		@param x ������
		@param y ������
		*/
		float Cof(int x, int y);
		float EigenvalueMax(float offset = 1e-8)const;
		/**
		@brief ���������ȡ�ľ���Ԫ��
		*/
		float RandSample();
		/**
		@brief ���ؾ���Ԫ�غ�
		@param num ���ô���
		@param _abs �Ƿ�ȡ����ֵ
		*/
		float sum(int num = 1, bool _abs = false)const;
		/**
		@brief ����ƽ��ֵ
		*/
		float mean()const;
		/**
		@brief ���������+
		��ӦԪ�����
		*/
		const Matrix operator + (const float val)const;
		/**
		@brief ���������+
		��ӦԪ�����
		*/
		const Matrix operator + (const Matrix &a)const;
		/**
		@brief ���������+=
		��Ԫ�����
		*/
		void operator += (const float val);
		/**
		@brief ���������+=
		��Ԫ�����
		*/
		void operator += (const Matrix &a);
		/**
		@brief ��Ԫ���������+
		��Ԫ�����
		*/
		friend const Matrix operator + (const float value, const Matrix &mat);
		/**
		@brief ���������-
		��Ԫ��ȡ�෴��
		*/
		const Matrix operator - (void)const;
		/**
		@brief ���������-
		��Ԫ�����
		*/
		const Matrix operator - (const float val)const;
		/**
		@brief ���������-
		��ӦԪ�����
		*/
		const Matrix operator - (const Matrix &a)const;
		/**
		@brief ���������-=
		��Ԫ�����
		*/
		void operator -= (const float val);
		/**
		@brief ���������-=
		��ӦԪ�����
		*/
		void operator -= (const Matrix &a);
		/**
		@brief ��Ԫ���������-
		��Ԫ�����
		*/
		friend const Matrix operator - (const float value, const Matrix &mat);
		/**
		@brief ���������*
		��Ԫ�����
		*/
		const Matrix operator * (const float val)const;
		/**
		@brief ���������*
		��ӦԪ�����
		*/
		const Matrix operator * (const Matrix &a)const;
		/**
		@brief ���������*=
		��Ԫ�����
		*/
		void operator *= (const float val);
		/**
		@brief ���������*=
		��ӦԪ�����
		*/
		void operator *= (const Matrix &a);
		/**
		@brief ��Ԫ���������*
		��Ԫ�����
		*/
		friend const Matrix operator * (const float value, const Matrix &mat);
		/**
		@brief ���������/
		��Ԫ�����
		*/
		const Matrix operator / (const float val)const;
		/**
		@brief ���������/
		����˷�
		*/
		const Matrix operator / (const Matrix &a)const;
		/**
		@brief ���������/=
		��Ԫ�����
		*/
		void operator /= (const float val);
		/**
		@brief ���������/=
		��ӦԪ�����
		*/
		void operator /= (const Matrix &a);
		/**
		@brief ��Ԫ���������/
		��Ԫ�����
		*/
		friend const Matrix operator / (const float value, const Matrix &mat);
		/**
		@brief ���������=
		���
		*/
		Matrix & operator = (const Matrix &temp);
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
		float& operator () (const int index)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param row ������
		@param col ������
		*/
		float& operator () (const int row, const int col)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param row ������
		@param col ������
		@param depth ͨ������
		*/
		float& operator () (const int row, const int col, const int depth)const;
		/**
		@brief ���ؾ����Ӧ�������л���
		@param index ����
		@param rc ������ʽ
		*/
		const Matrix operator () (const int index, X_Y_Z rc)const;
		/**
		@brief ���ؾ����Ӧ�������л���
		@param index ����
		@param rc ������ʽ
		*/
		const Matrix operator () (const int v1, const int v2, X_Y_Z rc)const;
		operator float *() {
			return matrix;
		}

		operator const float *() const {
			return matrix;
		}
		/**
		@brief ���ؾ����Ӧͨ������
		@param channel ͨ������
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
		@brief ����row*col*channel�ķ���
		@param row ��������
		@param col ��������
		@param depth ����ͨ����
		*/
		Mat_(int row, int col = 1, int channel = 1) : Mat(row, col, channel) {}
		/**
		@brief ����size_[0]*size_[1]*size_[2]�ķ���
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