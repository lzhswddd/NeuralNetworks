#include <iostream>
#include <fstream>
#include <windows.h>
#include "include.h"
#include "mtcnn.h"
#include "json.hpp"

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

void testNet()
{
	Net net;
	net.add(Conv2D(10, 3, false, ReLU, "conv_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, ReLU, "conv_2"));
	net.add(Conv2D(32, 3, false, ReLU, "conv_3")); 
	net.add(Conv2D(2, 1, false, Softmax, "fcl"));
	net.add(Loss(CROSSENTROPY, 1, "fc_loss"));
	net.add(Conv2D(4, 1, false, 0, "bbox"), "fcl", true);
	net.add(Loss(L2, 1, "bbox_loss"), "bbox");
	net.add(Conv2D(10, 1, false, 0, "landmark"), "fcl", true);
	net.add(Loss(L2, 1, "landmark_loss"), "landmark");
	net.initialize(Size3(12, 12, 3));
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
void testword(Net &net)
{
	vector<string> files;
	getFiles("F:\\study\\train\\images", files);//保存所有文件路径
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
	printf("样本正确率 %0.2lf%%\n", float(success) / float(sum) * 100);
}
void trainWordNet(Net &net)
{
	Size3 size(16, 16, 1);
	net.initialize(size);
	cout << net << endl;
	Mat x = mRand(-1, 1, size, true);
	vector<Mat> output = net(x);
	cout << "input " << x.size3() << endl;
	cout << "output ";
	for (Mat &y : output)
		cout << y.size3() << endl;

	int epoch = 20;
	TrainData trainData("F:\\study\\train", "data.txt", "images", 128);
	trainData.register_process(reverse, normalization);
	cout << "load traindata..." << endl;
	trainData.load_all_data(true);
	printf("start train net...\n");
	TrainOption option = { &trainData, 10, 1, true, false, nullptr, false, 0 };

	Train train;
	//train.regularization = true;
	//train.lambda = 0.01f;
	train.Fit(&net, OptimizerInfo(Adam, 0.001f), &option);
	net.save("./model/net");
	net.clear();
	net.load("./model/net");
	testword(net);
}
Net NN()
{
	Net net;
	net.add(Dense(50));
	net.add(BatchNorm());
	net.add(Activation(ReLU));
	net.add(Dense(50));
	net.add(BatchNorm());
	net.add(Activation(ReLU));
	net.add(Dense(50));
	net.add(BatchNorm());
	net.add(Activation(ReLU));
	net.add(Dense(23, Sigmoid));
	net.add(Loss(L2));
	return net;
}
Net CNN()
{
	Net net;
	net.add(Conv2D(10, 3, false, 0, "conv_1"));
	net.add(BatchNorm("bn_1"));
	net.add(Activation(ReLU, "activate_1"));
	//net.add(PReLU("activate_1"));
	net.add(MaxPool(Size(2, 2), 2, "maxpool_1"));
	net.add(Conv2D(16, 3, false, 0, "conv_2"));
	net.add(BatchNorm("bn_2"));
	net.add(Activation(ReLU, "activate_2"));
	//net.add(PReLU("activate_2"));
	net.add(Conv2D(32, 3, false, 0, "conv_3"));
	net.add(BatchNorm("bn_3"));
	net.add(Activation(ReLU, "activate_3"));
	net.add(Dense(23));
	//net.add(PReLU("activate_3"));
	//net.add(Conv2D(23, 3, false, 0, "conv_4"));
	//net.add(Reshape(Size3(23, 1, 1), "output"));
	net.add(Loss(SoftmaxCrossEntropy));
	return net;
}

int main()
{
	Srandom();
	//testTrain();
	//testNet(); 
	trainWordNet(CNN());
	//MTCNN mtcnn;
	//mtcnn.trainPNet(
	//	"D:\\mtcnn_data\\data\\seanlx",
	//	"train_12.txt", "mtcnn",
	//	256, false);
	/*vector<string> files;
	getFiles("E:\\deeplearn\\data\\kaggle_cifar10\\train", files);
	int count = 0;
	ofstream out("./negative.txt");
	for (string file : files)
	{
		Image img = Imread(file);
		resize(img, img, Size(12, 12), EqualIntervalSampling);
		Imwrite("D:\\mtcnn_data\\data\\dataset\\images\\12\\negative\\" + std::to_string(count+1) + ".jpg", img);
		count += 1;
		out << "12/negative/" + std::to_string(count + 1) + ".jpg " << 0 << endl;
		if (count > 10000)break;
	}out.close();*/
	pause();
	return 0;
}