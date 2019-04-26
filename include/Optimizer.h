#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

#include "NetParam.h"
#include "Train.h"
#include <vector>
#include <string>
using std::vector;
using std::string;

namespace nn {
	/**
	@brief Optimizer�Ż�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����
	*/
	class Optimizer
	{
	public:
		explicit Optimizer();
		Optimizer(float step);
		~Optimizer();
		/**
		@brief ��ʼ���Ż�������
		@param size �Ż�����ߴ�
		*/
		virtual void init(vector<Size3>& size) = 0;
		virtual void save(string file)const = 0;
		virtual void load(string file) = 0;
		/**
		@brief �����Ż�������1��
		@param x ��������
		@param y �������
		@param da �仯��
		@param size
		@param size
		*/
		virtual void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error) = 0;
		void RegisterTrain(Train *train);
		bool Enable(const Mat &x, const vector<Mat> &y)const;
		void RegisterMethod(OptimizerMethod method);
		OptimizerMethod OpMethod()const;
		virtual void copyTo(Optimizer* op)const = 0;
		virtual Mat Params()const = 0;
		float& Step() { return step; }

		static Optimizer* CreateOptimizer(Optimizer* op);
		static Optimizer* CreateOptimizer(OptimizerInfo optimizer_info);
		/**
		@brief CreateOptimizer �����Ż���
		�����Ż���
		@param opm �Ż�������
		@param step ѧϰ��
		@param loss_f ��ʧ����
		@param value ����
		*/
		static Optimizer* CreateOptimizer(OptimizerMethod opm, float step = 1e-2f, const Mat &value = (Mat_(3, 1) << 0.9f, 0.999e-3f, 1e-7f));
		static void SaveParam(FILE * file, const vector<Mat> *vec); 
		static void LoadParam(FILE * file, vector<Mat> *vec);
	protected:
		//����ʵ��
		Train * train;
		//ѧϰ��
		float step;
		//�̳л��������
		OptimizerMethod method;
	};
	/**
	@brief Method����ĺ�������
	ע��ģ�͵���ʧ����loss
	���ṩ�Ż�����
	*/
	class Method :public Optimizer
	{
	public:
		explicit Method();
		Method(float step);
		void init(vector<Size3>& size) {}
		void save(string file)const {}
		void load(string file) {}
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		Optimizer* minimize()const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	};
	/**
	@brief GradientDescentOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����GradientDescent
	a = a - step * df(a, x)
	*/
	class GradientDescentOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		*/
		explicit GradientDescentOptimizer(float step = 1e-2);
		void init(vector<Size3>& size) {}
		void save(string file)const {}
		void load(string file) {}
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		*/
		Optimizer* minimize()const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	};
	/**
	@brief MomentumOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����MomentumGradientDescent
	ma = momentum*ma + step * df(a, x)
	a = a - ma
	*/
	class MomentumOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param gama ����ϵ��
		*/
		explicit MomentumOptimizer(float step = 1e-2f);
		void init(vector<Size3>& size);
		void save(string file)const;
		void load(string file);
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(float momentum = 0.9f)const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	private:
		vector<Mat> ma;
		float momentum;
	};
	/**
	@brief NesterovMomentumOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����NesterovMomentumGradientDescent
	ma = momentum*ma + step * df(a - momentum*ma, x)
	a = a - ma
	*/
	class NesterovMomentumOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param gama ����ϵ��
		*/
		explicit NesterovMomentumOptimizer(float step = 1e-2);
		void init(vector<Size3>& size);
		void save(string file)const;
		void load(string file);
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(float momentum = 0.9f)const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	private:
		vector<Mat> ma;
		float momentum;
	};
	/**
	@brief AdagradOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����AdagradGradientDescent
	alpha = alpha + df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	class AdagradOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit AdagradOptimizer(float step = 1e-2f);
		void init(vector<Size3>& size);
		void save(string file)const;
		void load(string file);
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(float epsilon = 1e-7f)const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	private:
		vector<Mat> alpha;
		float epsilon;
	};
	/**
	@brief RMSPropOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����RMSPropGradientDescent
	alpha = beta*alpha + (1 - beta)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	class RMSPropOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param decay ����Ӧ����
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit RMSPropOptimizer(float step = 1e-2f);
		void init(vector<Size3>& size);
		void save(string file)const;
		void load(string file);
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(float decay = 0.9f, float epsilon = 1e-7f)const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	private:
		vector<Mat> alpha;
		float decay;
		float epsilon;
	};
	/**
	@brief AdamOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����AdamGradientDescent
	ma = beta1*ma + (1 - beta1)*df(a, x)
	alpha = beta2*alpha + (1 - beta2)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	class AdamOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param beta1 ����ϵ��
		@param beta2 ����Ӧϵ��
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit AdamOptimizer(float step = 1e-2f);
		void init(vector<Size3>& size);
		void save(string file)const;
		void load(string file);
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-7f)const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	private:
		vector<Mat> ma;
		vector<Mat> alpha;
		float epsilon;
		float beta1;
		float beta2;
	};
	/**
	@brief NesterovAdamOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����NesterovAdamGradientDescent
	ma = beta1*ma + (1 - beta1)*df(a - step/sqrt(alpha + epsilon)*ma, x)
	alpha = beta2*alpha + (1 - beta2)*df(a - step/sqrt(alpha + epsilon)*ma, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	class NesterovAdamOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param beta1 ����ϵ��
		@param beta2 ����Ӧϵ��
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit NesterovAdamOptimizer(float step = 1e-2f);
		void init(vector<Size3>& size);
		void save(string file)const;
		void load(string file);
		void Run(vector<Mat> &dlayer, const NetData *x, vector<float> &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-7f)const;
		Mat data(vector<string> &value_name)const;
		void copyTo(Optimizer* op)const;
		Mat Params()const;
	private:
		vector<Mat> ma;
		vector<Mat> alpha;
		float epsilon;
		float beta1;
		float beta2;
	};
	typedef Optimizer* OptimizerP;

}
#endif //__OPTIMIZER_H__
