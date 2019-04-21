#include <iostream>
#include <fstream>
#include "include.h"

using namespace nn;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

void reverse(const Image& src, Image&dst)
{
	if (src.channels != 1)
		RGB2Gray(src, dst);
	else
		dst = src;
	dst = 255 - src;
}

void normalization(const Mat& src, Mat&dst)
{
	dst = src / src.findmax();
}

const vector<Mat> label(const Mat &label)
{
	return vector<Mat>(1, value((float)label.maxAt(), 1, 1, 1));
}

void testTrain()
{
	TrainData traindata("./train", "data.txt", "images", 100, label);
	traindata.register_process(reverse, normalization);
	traindata.load_all_data();
	for (int &v : traindata.range) {
		const vector<NetData> *netData = traindata.batches();
		for (const NetData &data : *netData) {
			cout << data.input.Sum() << " " << data.label[0] << endl;
		}
	}
}
void testNet()
{
	Net net;
	net.add(Conv2D(10, 3, false, ReLU, "conv_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, ReLU, "conv_2"));
	net.add(Conv2D(32, 3, false, ReLU, "conv_3")); 
	net.add(Conv2D(2, 1, false, Softmax, "fcl"));
	net.add(Loss(CROSSENTROPY, "fc_loss"));
	net.add(Conv2D(4, 1, false, 0, "bbox"), "fcl", true);
	net.add(Loss(L2, "bbox_loss"), "bbox");
	net.add(Conv2D(10, 1, false, 0, "landmark"), "fcl", true);
	net.add(Loss(L2, "landmark_loss"), "landmark");
	net.initialize(3);
	cout << net << endl;
	Mat x = mRand(-1, 1, 12, 12, 3, true);
	vector<Mat> output = net(x);
	for(Mat &y : output)
		cout << y << endl;
}
const Mat ReadImageData(const Image &input)
{
	Image dst = 255 - input;//黑白转换
	Mat mat = Image2Mat(dst);//图像转为矩阵
	mat /= Max(mat);//归一化
	return mat;
}
void testTrainNet()
{
	Size3 size(16, 16, 1);
	Net net;
	net.add(Conv2D(10, 3, false, ReLU, "conv_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, ReLU, "conv_2"));
	net.add(Conv2D(32, 3, false, ReLU, "conv_3"));
	net.add(Dense(23, 0, "output"));
	net.add(Loss(SoftmaxCrossEntropy, "loss"));
	net.initialize(size);
	cout << net << endl;
	Mat x = mRand(-1, 1, size, true);
	vector<Mat> output = net(x);
	cout << "input " << x.size3() << endl;
	cout << "output ";
	for (Mat &y : output)
		cout << y.size3() << endl;

	int epoch = 20;
	TrainData traindata("./train", "data.txt", "images", 100);
	traindata.register_process(reverse, normalization);
	cout << "load traindata..." << endl;
	traindata.load_all_data();
	printf("start train net...\n");
	Train train;
	train.Fit(&net, traindata, OptimizerInfo(Adam, 1e-2f), epoch, 1, true);
	vector<string> files;
	getFiles("./train/images", files);//保存所有文件路径
	if (files.empty()) {
		printf("file is empty!\n");
		return;
	}
	string word = "ABCDEFGHJKLMNPRSTUVWXYZ";//所有结果
	int success = 0;
	int sum = 0;
	for (string &file : files) {
		size_t right = file.rfind('\\');
		size_t left = file.rfind('\\', right - 1) + 1;
		char key = file.substr(left, right - left)[0];
		Image image = Imread(file.c_str(), true);//用灰度读取图像
		vector<Mat> output = net(ReadImageData(image));//运行网络输出
		int adr = output[0].maxAt();//取最大值索引
		if (word[adr] == key)
			success += 1;
		sum += 1;
		cout << word[adr] << ' ' << adr << ' ' << key << '\t';
		cout << file << endl;
	}
	printf("样本正确率 %lf\n", float(success) / float(sum));
}

const Mat FaceClassifyLoss(const Mat &y, const Mat &y0)
{
	Mat y1 = zeros(y0.size3());
	if (y(0) == -1.0f)
		return y1;
	else {
		y1(int(y(0))) = 1;
		return CrossEntropy(y1, y0);
	}
}

const Mat FaceClassifyLoss_D(const Mat &y, const Mat &y0)
{
	Mat y1 = zeros(y0.size3());
	if (y(0) == -1.0f)
		return y1;
	else {
		y1(int(y(0))) = 1;
		return D_CrossEntropy(y1, y0);
	}
}

void image_processing(const Mat& src, Mat&dst)
{
	dst = (src - 127.5f)*0.0078125f;
}

const vector<Mat> label_processing(const Mat &label)
{
	vector<Mat> label_(2, value(label(0), 1, 1, 1));
	if (label(0) == 0.0f)
		label_[1] = zeros(1, 1, 4);
	else {
		label_[1] = Reshape(Block(label, 1, label.rows() - 1, 0, 0), Size3(1, 1, 4));
	}
	return label_;
}

void testPNet()
{
	Size3 size(12, 12, 3);
	Net net;
	net.add(Conv2D(10, 3, false, ReLU, "conv_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, ReLU, "conv_2"));
	net.add(Conv2D(32, 3, false, ReLU, "conv_3"));
	net.add(Conv2D(2, 1, false, Softmax, "fcl"));
	net.add(Loss(FaceClassifyLoss, FaceClassifyLoss_D, true, "fc_loss"));
	net.add(Conv2D(4, 1, false, 0, "bbox"), "fcl", true);
	net.add(Loss(L2, "bbox_loss"), "bbox");
	net.initialize(size);
	cout << net << endl;
	Mat x = mRand(-1, 1, size, true);
	vector<Mat> output = net(x);
	cout << "input " << x.size3() << endl;
	cout << "output ";
	for (Mat &y : output)
		cout << y.size3() << endl;
	//TrainData trainData("D:\\mtcnn_data\\data\\dataset", "imglists/train_12.txt", "images", 2048, label_processing);
	//trainData.register_process(0, image_processing);
	//for (int &v : trainData.range) {
	//	const vector<NetData> *netData = trainData.batches();
	//	for (const NetData &data : *netData) {
	//		cout << data.input.Sum() << " " << data.label[0] << " " << data.label[1].t() << endl;
	//	}
	//}

	Train train(&net);
	train.RegisterOptimizer(CreateOptimizer(GradientDescent, 1e-2f));

	TrainData trainData("./train/face", "train.txt", "images", 3, label_processing);
	trainData.register_process(0, image_processing);
	for (int &v : trainData.range) {
		const vector<NetData> *netData = trainData.batches();
		for (const NetData &data : *netData) {
			cout << data.input.Sum() << " " << data.label[0] << " " << data.label[1].t() << endl;
		}
		train.Fit(*netData);
	}

}
int main()
{
	Srandom();
	//testNet(); 
	//testTrainNet();
	testPNet();
	pause();
	return 0;
}