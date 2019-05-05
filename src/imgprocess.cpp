#include "imgprocess.h"
#include "alignMalloc.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define TJE_IMPLEMENTATION
#include "tiny_jpeg.h"

using std::vector;
using namespace nn;

const Mat nn::Image2Mat(const Image & src)
{
	if (src.empty())return Mat();
	Mat mat(src.rows, src.cols, src.channels);
	for (int i = 0; i < mat.rows(); ++i)
		for (int j = 0; j < mat.cols(); ++j)
			for (int z = 0; z < mat.channels(); ++z)
				mat(i, j, z) = (float)src(i, j, z);
	return mat;
}

const Image nn::Mat2Image(const Mat & src)
{
	if (src.empty())return Image();
	Image image(src.rows(), src.cols(), src.channels());
	for (int i = 0; i < src.rows(); ++i)
		for (int j = 0; j < src.cols(); ++j)
			for (int z = 0; z < src.channels(); ++z)
				image(i, j, z) = (uchar)src(i, j, z);
	return image;
}

const Image nn::Imread(std::string image_path, bool is_gray)
{
	return Imread(image_path.c_str(), is_gray);
}

const Image nn::Imread(const char * image_path, bool is_gray)
{
	int cols, rows, channels;
	uchar* ptr = stbi_load(image_path, &cols, &rows, &channels, 0);
	if (ptr == nullptr) {
		return Image();
		//fprintf(stderr, "load %s fail.\n", image_path);
		//throw "load image fail";
	}
	Image image(ptr, rows, cols, channels, cols, true);
	delete[]ptr;
	if (is_gray) {
		RGB2Gray(image, image);
	}
	return image;
}

const Mat nn::mImread(std::string image_path, bool is_gray)
{
	return Image2Mat(Imread(image_path, is_gray));
}

const Mat nn::mImread(const char * image_path, bool is_gray)
{
	return Image2Mat(Imread(image_path, is_gray));
}

void nn::mImwrite(std::string image_path, const Mat & image)
{
	Imwrite(image_path.c_str(), Mat2Image(image));
}

void nn::mImwrite(const char * image_path, const Mat & image)
{
	Imwrite(image_path, Mat2Image(image));
}

void nn::Imwrite(std::string image_path, const Image & image)
{
	Imwrite(image_path.c_str(), image);
}

void nn::Imwrite(const char * image_path, const Image & image)
{
	if (!tje_encode_to_file(image_path, image.cols, image.rows, image.channels, true, image.data)) {
		fprintf(stderr, "save %s fail.\n", image_path);
		throw std::exception("save image fail");
	}
}

void nn::RGB2Gray(const Image & src, Image & dst)
{
	if (src.empty())return;
	if (src.channels != 1) {
		Image img(src.rows, src.cols);
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				Vec<uchar> rgb = src(i, j);
				img(i, j, 0) = (uchar)(((int)rgb[0] * 30 + (int)rgb[1] * 59 + (int)rgb[2] * 11 + 50) / 100);
			}
		}
		dst = img;
	}
	else {
		src.copyTo(dst);
	}
}

void nn::resize(const Image & src, Image & dst, float xRatio, float yRatio, ReductionMothed mothed)
{
	if (src.empty())return;
	int rows = static_cast<int>(src.rows * yRatio);
	int cols = static_cast<int>(src.cols * xRatio);
	Image img(rows, cols, src.channels);
	switch (mothed)
	{
	case nn::EqualIntervalSampling:
		for (int i = 0; i < rows; i++) {
			int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
			for (int j = 0; j < cols; j++) {
				int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
				img(i, j) = src(row, col); //取得采样像素
			}
		}
		break;
	case nn::LocalMean:
	{
		int lastRow = 0;
		int lastCol = 0;

		for (int i = 0; i < rows; i++) {
			int row = static_cast<int>((i + 1) / yRatio + 0.5) - 1;
			for (int j = 0; j < cols; j++) {
				int col = static_cast<int>((j + 1) / xRatio + 0.5) - 1;
				Vec<uchar> temp;
				for (int idx = lastCol; idx <= col; idx++) {
					for (int jdx = lastRow; jdx <= row; jdx++) {
						temp[0] += src(jdx, idx)[0];
						temp[1] += src(jdx, idx)[1];
						temp[2] += src(jdx, idx)[2];
					}
				}

				int count = (col - lastCol + 1) * (row - lastRow + 1);
				img(i, j)[0] = temp[0] / count;
				img(i, j)[1] = temp[1] / count;
				img(i, j)[2] = temp[2] / count;

				lastCol = col + 1; //下一个子块左上角的列坐标，行坐标不变
			}
			lastCol = 0; //子块的左上角列坐标，从0开始
			lastRow = row + 1; //子块的左上角行坐标
		}
	}
	break;
	default:
		break;
	}
	dst = img;
}

void nn::resize(const Image & src, Image & dst, Size newSize, ReductionMothed mothed)
{
	resize(src, dst, newSize.w / float(src.cols), newSize.h / float(src.rows), mothed);
}

void nn::rotate(const Image & src, Image & dst, RotateAngle dice)
{
	switch (dice)
	{
	case nn::ROTATE_90_ANGLE:
	{
		Image img(src.cols, src.rows, src.channels);
		for (int row = 0; row < src.rows; ++row)
			for (int col = 0; col < src.cols; ++col)
				img(col, src.rows - 1 - row) = src(row, col);
		dst = img;
	}
		break;
	case nn::ROTATE_180_ANGLE:
	{
		Image img(src.rows, src.cols, src.channels);
		for (int row = 0, y = src.rows - 1; row < src.rows && y >=0; ++row, --y)
			for (int col = 0, x = src.cols - 1; col < src.cols && x >= 0; ++col, --x)
				img(row, col) = src(y, x);
		dst = img;
	}
		break;
	case nn::ROTATE_270_ANGLE:
	{
		Image img(src.cols, src.rows, src.channels);
		for (int row = 0; row < src.rows; ++row)
			for (int col = 0; col < src.cols; ++col)
				img(src.cols - 1 - col, row) = src(row, col);
		dst = img;
	}
		break;
	default:
		break;
	}
}

void nn::circle(Image & src, Point p, int radius, Color color, int lineWidth, bool fill)
{
	BresenhamCircle(src, p, radius, color, lineWidth, fill);
}

void nn::circle(Image & src, int x, int y, int radius, Color color, int lineWidth, bool fill)
{
	BresenhamCircle(src, Point(x, y), radius, color, lineWidth, fill);
}

void nn::SetPixel(Image & src, Point point, int x, int y, Color color)
{
	src(CHECK_INDEX(point.y + y, src.rows), CHECK_INDEX(point.x + x, src.cols)) = color;
	src(CHECK_INDEX(point.y + -y, src.rows), CHECK_INDEX(point.x + x, src.cols)) = color;
	src(CHECK_INDEX(point.y + y, src.rows), CHECK_INDEX(point.x + -x, src.cols)) = color;
	src(CHECK_INDEX(point.y + -y, src.rows), CHECK_INDEX(point.x + -x, src.cols)) = color;
	src(CHECK_INDEX(point.y + x, src.rows), CHECK_INDEX(point.x + y, src.cols)) = color;
	src(CHECK_INDEX(point.y + -x, src.rows), CHECK_INDEX(point.x + y, src.cols)) = color;
	src(CHECK_INDEX(point.y + x, src.rows), CHECK_INDEX(point.x + -y, src.cols)) = color;
	src(CHECK_INDEX(point.y + -x, src.rows), CHECK_INDEX(point.x + -y, src.cols)) = color;
}

void addContour(vector<Point> &contour, Point point, int x, int y, int rows, int cols)
{
	contour.push_back(Point(CHECK_INDEX(point.y + y, rows), CHECK_INDEX(point.x + x, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + -y, rows), CHECK_INDEX(point.x + x, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + y, rows), CHECK_INDEX(point.x + -x, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + -y, rows), CHECK_INDEX(point.x + -x, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + x, rows), CHECK_INDEX(point.x + y, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + -x, rows), CHECK_INDEX(point.x + y, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + x, rows), CHECK_INDEX(point.x + -y, cols)));
	contour.push_back(Point(CHECK_INDEX(point.y + -x, rows), CHECK_INDEX(point.x + -y, cols)));
}

void drawFill(Image & src, vector<Point> &contour, Color color)
{

}

void nn::BresenhamCircle(Image & src, Point point, int radius, Color color, int lineWidth, bool fill)
{
	vector<Point> contour;
	for(int wid = -lineWidth/2 ; wid <= lineWidth/2;++wid)
		for (int x = 0, y = radius + wid, p = 3 - 2 * (radius + wid); x <= y; x++) {
			SetPixel(src, point, x, y, color);
			if (fill)
				addContour(contour, point, x, y, src.rows, src.cols);
			if (p >= 0) {
				p += 4 * (x - y) + 10;
				y--;
			}
			else {
				p += 4 * x + 6;
			}
		}
}

void nn::rectangle(Image & src, int x1, int y1, int x2, int y2, Color color, int lineWidth, bool fill)
{
	for (int col = max(x1, 0); col <= min(src.cols - 1, x2); ++col)
		for (int wid = -lineWidth / 2; wid <= lineWidth / 2; ++wid)
			src(CHECK_INDEX(y1 + wid, src.rows - 1), CHECK_INDEX(col, src.cols - 1)) = color;
	for (int col = max(x1, 0); col <= min(src.cols - 1, x2); ++col)
		for (int wid = -lineWidth / 2; wid <= lineWidth / 2; ++wid)
			src(CHECK_INDEX(y2 + wid, src.rows - 1), CHECK_INDEX(col, src.cols - 1)) = color;
	for (int row = max(y1, 0); row <= min(src.rows - 1, y2); ++row)
		for (int wid = -lineWidth / 2; wid <= lineWidth / 2; ++wid)
			src(CHECK_INDEX(row + wid, src.rows - 1), CHECK_INDEX(x1 + wid, src.cols - 1)) = color;
	for (int row = max(y1, 0); row <= min(src.rows - 1, y2); ++row)
		for (int wid = -lineWidth / 2; wid <= lineWidth / 2; ++wid)
			src(CHECK_INDEX(row + wid, src.rows - 1), CHECK_INDEX(x2 + wid, src.cols - 1)) = color;
	if (fill)
	{
		for (int col = max(x1, 0); col <= min(src.cols - 1, x2); ++col)
			for (int row = max(y1, 0); row <= min(src.rows - 1, y2); ++row)
				src(row, col) = color;
	}
}

void nn::rectangle(Image & src, Rect rect, Color color, int lineWidth, bool fill)
{
	rectangle(src, rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, color, lineWidth, fill);
}

void nn::drawContours(Image & src, const std::vector<Point>& contours, int radius, Color color, int lineWidth, bool fill)
{
	for (const Point &p : contours) {
		circle(src, p, radius, color, fill);
	}
}

void nn::drawContours(Image & src, const std::vector<std::vector<Point>>& contours, int index, int radius, Color color, int lineWidth, bool fill)
{
	if (index == -1) {
		for (const std::vector<Point> &contour : contours)
			for (const Point &p : contour)
				circle(src, p, radius, color, lineWidth, fill);
	}
	else {
		for (const Point &p : contours[index])
			circle(src, p, radius, color, lineWidth, fill);
	}
}

void nn::projection(const Image& src, Mat &vertical, Mat &horizontal)
{
	if (src.empty())return;
	horizontal = zeros(1, src.cols, src.channels);
	vertical = zeros(src.rows, 1, src.channels);
	FOR_IMAGE(i, src, 1) {
		FOR_IMAGE(j, src, 2) {
			FOR_IMAGE(k, src, 3) {
				horizontal(0, j, k) += src(i, j, k);
				vertical(i, 0, k) += src(i, j, k);
			}
		}
	}
}

void nn::verticalProjection(const Image & src, Mat &vertical)
{
	if (src.empty())return;
	vertical = zeros(src.rows, 1, src.channels);
	FOR_IMAGE(i, src, 1) {
		FOR_IMAGE(k, src, 3) {
			int sum = 0;
			FOR_IMAGE(j, src, 2) {
				sum += src(i, j, k);
			}
			vertical(i, 0, k) = (float)sum;
		}
	}
}

void nn::horizontalProjection(const Image & src, Mat &horizontal)
{
	if (src.empty())return;
	horizontal = zeros(1, src.cols, src.channels);
	FOR_IMAGE(i, src, 2) {
		FOR_IMAGE(k, src, 3) {
			int sum = 0;
			FOR_IMAGE(j, src, 1) {
				sum += src(j, i, k);
			}
			horizontal(0, i, k) = (float)sum;
		}
	}
}

