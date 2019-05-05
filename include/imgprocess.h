#ifndef __IMGPROCESS_H__
#define __IMGPROCESS_H__

#include "image.h"
#include "vriable.h"
#include <vector>
#include <string>

#define CHECK_INDEX(v, limit) (min((max((v), 0)), (limit)))

namespace nn
{
	//Í¼Ïñ×ª¾ØÕó
	const Mat Image2Mat(const Image &src);
	//¾ØÕó×ªÍ¼Ïñ
	const Image Mat2Image(const Mat &src);
	//¶ÁÈ¡Í¼Ïñ
	const Image Imread(std::string image_path, bool is_gray = false);
	//¶ÁÈ¡Í¼Ïñ
	const Image Imread(const char *image_path, bool is_gray = false);
	//¶ÁÈ¡Í¼Ïñ
	const Mat mImread(std::string image_path, bool is_gray = false);
	//¶ÁÈ¡Í¼Ïñ
	const Mat mImread(const char *image_path, bool is_gray = false);
	//±£´æÍ¼Ïñ
	void mImwrite(std::string image_path, const Mat& image);
	//±£´æÍ¼Ïñ
	void mImwrite(const char *image_path, const Mat& image);
	//±£´æÍ¼Ïñ
	void Imwrite(std::string image_path, const Image& image);
	//±£´æÍ¼Ïñ
	void Imwrite(const char *image_path, const Image& image);
	//RGB×ª»Ò¶È
	void RGB2Gray(const Image& src, Image& dst);
	//°´±ÈÀýËõ·Å
	void resize(const Image & src, Image & dst, float xRatio, float yRatio, ReductionMothed mothed);
	//Ëõ·Å
	void resize(const Image& src, Image& dst, Size newSize, ReductionMothed mothed);
	
	void rotate(const Image& src, Image &dst, RotateAngle dice);

	void SetPixel(Image & src, Point point, int x, int y, Color color);
	void circle(Image &src, Point p, int radius, Color color, int lineWidth = 1, bool fill = false);
	void circle(Image &src, int x, int y, int radius, Color color, int lineWidth = 1, bool fill = false);
	void BresenhamCircle(Image &src, Point point, int radius, Color color, int lineWidth, bool fill = false);
	void rectangle(Image &src, int x1, int y1, int x2, int y2, Color color, int lineWidth = 1, bool fill = false);
	void rectangle(Image &src, Rect rect, Color color, int lineWidth = 1, bool fill = false);
	void drawContours(Image &src, const std::vector<Point> &contours, int radius, Color color, int lineWidth = 1, bool fill = false);
	void drawContours(Image &src, const std::vector<std::vector<Point>> &contours, int index, int radius, Color color, int lineWidth = 1, bool fill = false);

	void projection(const Image& src, Mat &vertical, Mat &horizontal);
	void verticalProjection(const Image& src, Mat &vertical);
	void horizontalProjection(const Image & src, Mat &horizontal);
}
#endif // !__IMGPROCESS_H__
