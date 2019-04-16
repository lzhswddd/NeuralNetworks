#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "Vriable.h"
#include "Mat.h"

namespace nn {	

	//π‹¿ÌÕºœÒ¿‡
	class Image
	{
	public:
		explicit Image();
		Image(Size3 size);
		Image(Size size, int channels = 1);
		Image(const char *image_path);
		Image(int rows, int cols, int channels = 1);
		Image(uchar *data, int rows, int cols, int channels);
		Image(const Image &src);
		~Image();
		int reCount()const;
		int length()const;
		bool empty()const;
		void copyTo(Image &src)const;
		void operator = (const Image &src);
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
		int cols;
		int rows;
		int channels;
		uchar *data;

	protected:
		int *count = nullptr;
		void copy(const Image &src);
	};

}

#endif //__IMAGE_H__