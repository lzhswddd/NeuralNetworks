#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "vriable.h"
#include "mat.h"
#include <string>

#define FOR_IMAGE(i, img, dire) for(int (i) = 0; (i) < ((dire) == 1?(img).rows:((dire) == 2 ? (img).cols : (img).channels)); ++i)

namespace nn {	

	//¹ÜÀíÍ¼ÏñÀà
	class Image
	{
	public:
		explicit Image();
		Image(Size3 size);
		Image(Size size, int channels = 1);
		Image(std::string image_path, bool is_gray = false);
		Image(int rows, int cols, int channels = 1);
		Image(uchar *data, int rows, int cols, int channels, int step, bool iscopy = false);
		Image(const Image &src);
		~Image();
		int reCount()const;
		int length()const;
		bool empty()const;
		void create(int rows, int cols);
		void create(int rows, int cols, int channels);
		void copyTo(Image &src)const;
		void release();
		Image ROI(Rect rect);
		Image clone()const;
		const Image operator + (const uchar value)const;
		const Image operator + (const Image &image)const;
		void operator += (const Image &image);
		void operator += (uchar value);
		friend const Image operator + (uchar value, const Image &src);
		const Image operator - (uchar value)const; 		
		const Image operator - (const Image &image)const;
		void operator -= (const Image &image);
		void operator -= (uchar value);
		friend const Image operator - (const uchar value, const Image &src);
		Vec<uchar> operator () (Point pos)const;
		Vec<uchar> operator () (int row, int col)const;
		Vec<uchar> operator () (int index)const;
		uchar& operator () (Point3i pos)const;
		uchar& operator () (int row, int col, int channel)const;
		Image operator () (Rect rect);
		void operator = (const Image &src);
		friend std::ostream & operator << (std::ostream &out, const Image &ma)
		{
			FOR_IMAGE(i, ma, 1) {
				FOR_IMAGE(j, ma, 2) {
					FOR_IMAGE(k, ma, 3) {
						out << (int)ma(i, j, k) << ' ';
					}
				}out << std::endl;
			}out << std::endl;
			return out;
		}
		int cols;
		int rows;
		int channels;
		int step;
		uchar *data;

	protected:
		int *count = nullptr;
		void copy(const Image &src);
	};

}

#endif //__IMAGE_H__