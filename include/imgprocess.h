#ifndef __IMGPROCESS_H__
#define __IMGPROCESS_H__

#include "Image.h"
#include <vector>
#include <string>

#define CHECK_INDEX(v, limit) (min((max((v), 0)), (limit)))

namespace nn
{
	/**
	EqualIntervalSampling 等间隔采样
	LocalMean 局部均值
	*/
	enum ReductionMothed
	{
		EqualIntervalSampling = 0,
		LocalMean
	};

	class Color
	{
	public:
		Color(uchar v) : r(v), g(v), b(v) {}
		Color(uchar r, uchar g, uchar b) : r(r), g(g), b(b){}
		uchar r;
		uchar g;
		uchar b;
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
	};

	//图像转矩阵
	const Mat Image2Mat(const Image &src);
	//矩阵转图像
	const Image Mat2Image(const Mat &src);
	//读取图像
	const Image Imread(std::string image_path, bool is_gray = false);
	//读取图像
	const Image Imread(const char *image_path, bool is_gray = false);
	//读取图像
	const Mat mImread(std::string image_path, bool is_gray = false);
	//读取图像
	const Mat mImread(const char *image_path, bool is_gray = false);
	//保存图像
	void Imwrite(std::string image_path, Image& image);
	//保存图像
	void Imwrite(const char *image_path, const Mat & image);
	//RGB转灰度
	void RGB2Gray(const Image& src, Image& dst);
	//按比例缩放
	void resize(const Image & src, Image & dst, double xRatio, double yRatio, ReductionMothed mothed);
	//缩放
	void resize(const Image& src, Image& dst, Size newSize, ReductionMothed mothed);
	
	void SetPixel(Image & src, Point point, int x, int y, Color color);
	void circle(Image &src, Point p, int radius, Color color, int lineWidth = 1, bool fill = false);
	void circle(Image &src, int x, int y, int radius, Color color, int lineWidth = 1, bool fill = false);
	void BresenhamCircle(Image &src, Point point, int radius, Color color, int lineWidth, bool fill = false);
	void rectangle(Image &src, int x1, int y1, int x2, int y2, Color color, int lineWidth = 1, bool fill = false);
	void rectangle(Image &src, Rect rect, Color color, int lineWidth = 1, bool fill = false);
	void drawContours(Image &src, const std::vector<Point> &contours, int radius, Color color, int lineWidth = 1, bool fill = false);
	void drawContours(Image &src, const std::vector<std::vector<Point>> &contours, int index, int radius, Color color, int lineWidth = 1, bool fill = false);
}
#endif // !__IMGPROCESS_H__
