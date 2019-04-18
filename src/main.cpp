#include <iostream>
#include <fstream>
#include <time.h>

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

void reverse(const Image& src, Image&dst)
{
	dst = 255 - src;
}

void normalization(const Mat& src, Mat&dst)
{
	dst = src / src.findmax();
}

const Mat label(const Mat &label)
{
	return Mat(label.maxAt());
}

void testTrain()
{
	TrainData traindata("./train", "data.txt", "images", 100, label);
	traindata.register_process(reverse, normalization);
	Mat x, y;
	for (int &v : traindata.range) {
		traindata.batch(x, y);
		cout << x.Sum() << " " << y << endl;
	}
	traindata.reset();
	for (int &v : traindata.range) {
		traindata.batch(x, y);
		cout << x.Sum() << " " << y << endl;
	}
}

int main()
{
	Image img("F:\\task\\FittingCircle\\images\\5.png");
	if (img.empty())return -1;
	Image image;
	RGB2Gray(img, image);
	image = 255 - image;
	Image a;
	rotate(img, a, ROTATE_90_ANGLE);
	Imwrite("C:\\Users\\lzh\\Desktop\\1.bmp", a);
	Mat v, h;
	projection(image, v, h);
	Rect rect;
	bool black = false;
	for (int i = 0; i < v.length(); ++i) {
		if (!black && v(i) != 0) {
			black = true;
			rect.y = i;
		}
		else if(black && v(i) == 0) {
			rect.height = i - rect.y;
			break;
		}
	} 
	if (black)
		rect.height = image.rows;
	black = false;
	for (int i = 0; i < h.length(); ++i) {
		if (!black && h(i) != 0) {
			black = true;
			rect.x = i;
		}
		else if (black && h(i) == 0) {
			rect.width = i - rect.x;
			break;
		}
	}
	rect.x += 5;
	rect.y += 1;
	rect.width -= 5;
	rect.height -= 5;
	Image roi = img(rect).clone();
	//rectangle(roi, Rect(50, 50, 500, 500), Color(0), 10, true);
	Imwrite("C:\\Users\\lzh\\Desktop\\2.bmp", roi);
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