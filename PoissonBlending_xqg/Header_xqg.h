#ifndef _HEADER_H
#define _HEADER_H

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include "Eigen/Sparse" 
#include <time.h>

#include <vector>

#define ATF at<float>
#define ATU at<uchar>
using namespace cv;
using namespace std;

typedef Eigen::SparseMatrix<float> SpMat;
typedef Eigen::Triplet<float> T;

SpMat getA(Mat & Mask, int** Index, unsigned int unknownSize);

Eigen::VectorXf getB_normal (Mat& src, Mat &dest, int **Index, Mat & Mask); //normal
Eigen::VectorXf getB_mixture(Mat& src, Mat &dest, int **Index, Mat & Mask); // mixture

void seamlessClone_xqg(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags);


#endif

