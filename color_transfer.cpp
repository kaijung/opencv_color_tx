#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

void colorTransfer(const Mat& src, Mat& dst) {
	Mat labsrc, labdst;

	cvtColor(src, labsrc, COLOR_BGR2Lab);
	cvtColor(dst, labdst, COLOR_BGR2Lab);

	labsrc.convertTo(labsrc, CV_32FC3);
	labdst.convertTo(labdst, CV_32FC3);

	//计算三个通道的均值与方差
	Scalar meansrc, stdsrc, meandst, stddst;
	meanStdDev(labsrc, meansrc, stdsrc);
	meanStdDev(labdst, meandst, stddst);

	//三通道分离
	vector<Mat> /*srcsplit,*/ dstsplit;
    // split(labsrc, srcsplit);
	split(labdst, dstsplit);

	for (int i = 0; i < 3; i++) {
       dstsplit[i] -= meandst[i];
       dstsplit[i] *= (stddst[i] / stdsrc[0]);
       dstsplit[i] += meansrc[i];
    }

	//合并每个通道
	merge(dstsplit, dst);
    dst.convertTo(dst, CV_8UC1);

	//从lab空间转换到RGB空间
	cvtColor(dst, dst, COLOR_Lab2BGR);
}


int main(int argc, char** argv) {
	Mat src, dst;
	src = imread("images/ocean_day.jpg", CV_LOAD_IMAGE_COLOR);
	dst = imread("images/shoe.jpg", CV_LOAD_IMAGE_COLOR);

	colorTransfer(src, dst);
	imwrite("dst.jpg", dst);
    // imshow("src", src);
	imshow("dst", dst);
	waitKey(0);
	return 0;
}