#ifndef __MTCNN_H__
#define __MTCNN_H__

#include "Net.h"
#include "method.h"

namespace nn {
	class MTCNN
	{
	public:
		MTCNN();
		~MTCNN();

		void detect(Mat &img_, std::vector<Bbox> &finalBbox_);

		void PNet();

		void RNet();

		void ONet();

		Net Pnet, Rnet, Onet;
		Mat img;
		const float nms_threshold[3] = { 0.5f, 0.7f, 0.7f };

		const int MIN_DET_SIZE = 12;
		std::vector<Bbox> firstBbox_, secondBbox_, thirdBbox_;
		int img_w, img_h;
	private:
		const float threshold[3] = { 0.8f, 0.8f, 0.6f };
		int minsize = 40;
		const float pre_facetor = 0.709f;
	};
}

#endif //__MTCNN_H__