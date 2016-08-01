#ifndef CV_SEAMLESS_CLONING_HPP___
#define CV_SEAMLESS_CLONING_HPP___

#include "opencv2/photo/photo.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <time.h>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

enum
{
	NORMAL_CLONE = 1,
	MIXED_CLONE = 2,
	MONOCHROME_TRANSFER = 3
};

// void normalClone(const cv::Mat& destination, const cv::Mat &mask, const cv::Mat &wmask, cv::Mat &cloned, int flag);
void computeDerivatives(const Mat& destination, const Mat &patch, const Mat &binaryMask);
void computeGradientX(const Mat &img, Mat &gx);
void computeGradientY(const Mat &img, Mat &gy);
void computeLaplacianX(const Mat &img, Mat &laplacianX);
void computeLaplacianY(const Mat &img, Mat &laplacianY);
void dst(const Mat& src, Mat& dest, bool invert);
void idst(const Mat& src, Mat& dest);
void solve(const Mat &img, Mat& mod_diff, Mat &result);
void poissonSolver(const Mat &img, Mat &laplacianX, Mat &laplacianY, Mat &result);
void initVariables(const Mat &destination, const Mat &binaryMask);
void scalarProduct(Mat mat, float r, float g, float b);
void arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result);
void poisson(const Mat &destination);
void evaluate(const Mat &I, const Mat &wmask, const Mat &cloned);
void normalClone(const Mat &destination, const Mat &patch, const Mat &binaryMask, Mat &cloned, int flag);
void localColorChange(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float red_mul = 1.0,
	float green_mul = 1.0, float blue_mul = 1.0);

void illuminationChange(Mat &I, Mat &mask, Mat &wmask, Mat &cloned, float alpha, float beta);
void textureFlatten(Mat &I, Mat &mask, Mat &wmask, float low_threshold,
	float high_threshold, int kernel_size, Mat &cloned);


void seamlessClone(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags);
void colorChange(InputArray _src, InputArray _mask, OutputArray _dst, float r, float g, float b);
// void illuminationChange(InputArray _src, InputArray _mask, OutputArray _dst, float a, float b);
// void textureFlattening(InputArray _src, InputArray _mask, OutputArray _dst, float low_threshold, float high_threshold, int kernel_size);


#endif