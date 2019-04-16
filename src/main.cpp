#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <iomanip>
#include <io.h>
#include <direct.h>
#include <Windows.h>
#include "include.h"

using namespace nn;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

class Pnet: public Net
{
public:
	Pnet() :net(), fc(), bbox()
	{
		net.add(Conv2D(10, 3, false, ReLU));
		net.add(MaxPool(Size(2, 2), 2));
		net.add(Conv2D(16, 3, false, ReLU));
		net.add(Conv2D(32, 3, false, ReLU));
		fc.add(Conv2D(2, 1, false, Softmax));
		bbox.add(Conv2D(4, 1, false));
	}

	void initialize(int channel)
	{
		channel = net.initialize(channel);
		fc.initialize(channel);
		bbox.initialize(channel);
	}

	const vector<Mat> forward(const Mat& x)const
	{
		vector<Mat> output(2);
		Mat y = net(x);
		output[0] = fc(y);
		output[1] = bbox(y);
		return output;
	}

	const vector<Mat> operator()(const Mat & input) const
	{
		return forward(input);
	}

private:
	Net net;
	Net fc;
	Net bbox;
};

int main()
{
	Mat m = zeros(100, 100);
	Mat a[10000];
	int i = 0;
	while (1) {
		a[i] = m;
		i = (i + 1) % 10000;
	}
	/*Srandom();
	Pnet net;
	net.initialize(3);
	Mat x = mRand(-1, 1, 12, 12, 3, true);
	vector<Mat> y = net(x);
	cout << y[0] << endl;
	cout << y[1] << endl;
	pause();*/
	//Image image = Imread("E:\\study\\c++\\FittingCircle\\FittingCircle\\images\\1.png");
	////rectangle(image, Rect(50, 50, 500, 500), Color(255, 0, 0), 10, true);
	//circle(image, Point(1500, 1500), 100, Color(255, 0, 0), 10);
	//Imwrite("C:\\Users\\lzh\\Desktop\\test.bmp", image);
	return 0;
}