#ifndef __MTCNN_H__
#define __MTCNN_H__

#include "net.h"
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

		static const Mat FaceClassifyLoss(const Mat &y, const Mat &y0);
		static const Mat FaceClassifyLoss_D(const Mat &y, const Mat &y0);
		static const Mat BboxLoss(const Mat &y, const Mat &y0);
		static const Mat BboxLoss_D(const Mat &y, const Mat &y0);
		static void image_processing(const Mat& src, Mat&dst);
		static const vector<Mat> label_processing(const Mat &label);
		static Net create_pnet(bool load = false, string model = "");
		void trainPNet(string rootpath, string imglist, string imagedir,
			int batch_size, bool load = true, string model = "./net/net.param", string optimizer_param = "./net/optimizer.param");
		void testPNet(string model = "./net/net.param", string pic = "1.jpg", string savepath = "test.jpg");

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