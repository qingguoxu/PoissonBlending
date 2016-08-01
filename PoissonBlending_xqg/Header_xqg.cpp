#include "Header_xqg.h"


cv::Mat final_result;
int extra_size = 1; //7

void seamlessClone_xqg(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags)
{

	const Mat raw_src = _src.getMat();
	const Mat raw_dest = _dst.getMat();
	const Mat raw_mask = _mask.getMat();

	final_result = raw_dest;

	/////
	
	Mat gray = Mat(raw_mask.size(), CV_8UC1);

	if (raw_mask.channels() == 3)
		cvtColor(raw_mask, gray, COLOR_BGR2GRAY);
		
	else
		gray = raw_mask;

	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			if (gray.ATU(i, j) > 50)
				gray.ATU(i, j) = 255;
			else
				gray.ATU(i, j) = 0;
		}
	}


	int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
	int h = raw_mask.size().height;
	int w = raw_mask.size().width;

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (gray.ATU(i, j) == 255)
			{
				minx = std::min(minx, i);
				maxx = std::max(maxx, i);
				miny = std::min(miny, j);
				maxy = std::max(maxy, j);
			}
		}
	}

	int lenx = maxx - minx;
	int leny = maxy - miny;

	// cout << "( " << lenx / 2 << "," << leny / 2 << " )" << endl;

	int len_x = lenx + 2 * extra_size;
	int len_y = leny + 2 * extra_size;

	// cout << "( " << len_x / 2 << "," << len_y / 2 << " )" << endl;

	/*if ((minx + len_x) > raw_mask.rows || (miny+len_y)>raw_mask.cols)
	{
	cout << "Wrong mask size." << endl;
	exit(-1);
	}*/
	if ((minx + len_x) > raw_mask.rows) 
	{
		len_x = raw_mask.rows - minx;
	}

	if ((miny + len_y) > raw_mask.cols)
	{
		len_y = raw_mask.cols - miny;
	}
//	Mat patch = Mat::zeros(Size(leny, lenx), CV_8UC3);

	int minxd = (p.x - extra_size) > 0 ? (p.x - extra_size) : 0;
	int maxxd = (p.x + lenx + extra_size) > raw_dest.rows ? raw_dest.rows : (p.x + lenx + extra_size);
	int minyd = (p.y - extra_size) > 0 ? (p.y - extra_size) : 0;
	int maxyd = (p.y + leny + extra_size) > raw_dest.cols ? raw_dest.cols : (p.y + leny + extra_size);
	
	cout << "( " << minxd << "," << minyd << " )" << endl;
	cout << "( " << maxxd << "," << maxyd << " )" << endl;

	/*int minxd = (p.x - extra_size) > 0 ? (p.x - extra_size) : 0;
	int maxxd = (p.x + lenx + extra_size) > raw_dest.rows ? raw_dest.rows : (p.x + lenx + extra_size);
	int minyd = (p.y - extra_size) > 0 ? (p.y - extra_size) : 0;
	int maxyd = (p.y + leny + extra_size) > raw_dest.cols ? raw_dest.cols : (p.y + leny + extra_size);*/

	/*
	int len_xd = 0, len_yd = 0;
//	int len_xs = 0, len_ys = 0;

	len_xd = (maxxd - minxd) > raw_dest.rows ? raw_dest.rows : (maxxd - minxd);
	len_yd = (maxyd - minyd) > raw_dest.cols ? raw_dest.cols : (maxyd - minyd);

	
	if ((minx + len_xd) > raw_mask.cols)
	{
		cout << "wrong mask size" << endl;
		exit(-1);
	}
	*/
	Rect roi_d(minyd, minxd, len_y, len_x);
	Rect roi_s(miny-extra_size, minx-extra_size, len_y, len_x);

	Mat dest = raw_dest(roi_d);
	Mat src = raw_src(roi_s);
	Mat mask = raw_mask(roi_s);

	string folder = "../img/";

	/*imwrite(folder + "output.png", dest);
	imwrite(folder + "output1.png", src);
	imwrite(folder + "output2.png", mask);
	*/
//	Mat gray2 = Mat(mask.size(), CV_8UC1);
	Mat dst_mask = Mat::zeros(dest.size(), CV_8UC1);
	if (mask.channels() == 3)
		cvtColor(mask, dst_mask, COLOR_BGR2GRAY);
	else
		dst_mask = mask;
	
	for (int i = 0; i < dst_mask.rows; i++)
	{
		for (int j = 0; j < dst_mask.cols; j++)
		{
			if (dst_mask.ATU(i, j) > 50)
				dst_mask.ATU(i, j) = 255;
			else
				dst_mask.ATU(i, j) = 0;
		}
	}

	/* my own implementation*/

	//unsigned int unknownSize = getMaskNum(originalMask);
	unsigned int unknownSize = dest.cols*dest.rows;
	int ** Index = new  int *[dest.rows];

	for (int i = 0; i < dest.rows; i++)
	{
		Index[i] = new  int[dest.cols];

	}

	unsigned int counter = 0;
	// assign Index
	for (int i = 0; i < dest.rows; i++)
	{
		for (int j = 0; j < dest.cols; j++)
		{
			Index[i][j] = counter;
			counter++;
		}
	}


//	cout << " counter :" << counter << endl;

	// compute A
	SpMat A = getA(dst_mask, Index, unknownSize);

	vector<Mat> rgb_dst;
	vector<Mat> rgb_src;

	Mat src_float;
	Mat dst_float;
	src.convertTo(src_float, CV_32FC3);
	dest.convertTo(dst_float, CV_32FC3);

	split(src_float, rgb_src);
	split(dst_float, rgb_dst);


	for (int ch = 0; ch < 3; ch++)
	{

		Eigen::VectorXf b(unknownSize);
		if (flags == 1) // normal
		{
			b = getB_normal(rgb_src[ch], rgb_dst[ch], Index, dst_mask);
			
		}
		else if (flags == 2) // mixture
		{
			b = getB_mixture(rgb_src[ch], rgb_dst[ch], Index, dst_mask);
		}
	
		// solve Ax = b

		Eigen::SimplicialCholesky<SpMat> chol(A);
		Eigen::VectorXf x = chol.solve(b);

		for (int i = 0; i < dest.rows; i++)
		{
			for (int j = 0; j < dest.cols; j++)
			{
				rgb_dst[ch].ATF(i, j) = x[ Index[i][j] ];
			}
		}

	}


	Mat newResult;
	merge(rgb_dst, newResult);

	newResult.convertTo(newResult, CV_8UC3);
	newResult.copyTo(final_result(roi_d));
	
	imwrite(folder + "final_result.png", final_result);

	cv::namedWindow("Output");
	cv::moveWindow("Output", 10, 50);
	imshow("Output", final_result);
	
	

}


Eigen::VectorXf getB_normal(Mat& src, Mat &dest, int **Index, Mat & mask)
{
	/*Mat temp;
	dest.convertTo(temp, CV_8UC1);
	imshow("cropped", temp);
	waitKey(0);*/

	unsigned int unknownSize = dest.cols*dest.rows;
	Eigen::VectorXf b(unknownSize);

	// handle boundary cases
	
	int currentX_index;
	for (unsigned int i = 0; i < dest.rows; i++)
	{
		for (unsigned int j = 0; j < dest.cols; j++)
		{

			float current_value_dest = dest.ATF(i, j);
			currentX_index = Index[i][j];

			b(currentX_index) = 0;
			if (mask.ATU(i, j) == 0) // outside of the mask
			{
				b(currentX_index) = current_value_dest;
			}
			else // in the mask
			{

				float current_value = src.ATF(i, j);

				//i-1,j
				if ((i - 1) >= 0)
				{
					b(currentX_index) = b(currentX_index) + current_value - src.ATF(i - 1 , j );
					if (mask.ATU(i - 1 , j) == 0)
					{
						b(currentX_index) = b(currentX_index) + dest.ATF(i - 1, j);

					}

				}

				//i+1,j
				if ((i + 1 ) <= (dest.rows - 1))
				{
					b(currentX_index) = b(currentX_index) + current_value - src.ATF(i + 1 , j );
					if (mask.ATU(i + 1, j ) == 0)
					{
						b(currentX_index) = b(currentX_index) + dest.ATF(i + 1, j);

					}

				}

				//i,j-1
				if ((j - 1 ) >= 0)
				{
					b(currentX_index) = b(currentX_index) + current_value - src.ATF(i , j - 1 );
					if (mask.ATU(i , j - 1) == 0)
					{
						b(currentX_index) = b(currentX_index) + dest.ATF(i, j - 1);

					}

				}
				//i,j+1
				if ((j + 1 ) <= (dest.cols - 1))
				{
					b(currentX_index) = b(currentX_index) + current_value - src.ATF(i , j + 1 );
					if (mask.ATU(i , j + 1 ) == 0)
					{
						b(currentX_index) = b(currentX_index) + dest.ATF(i, j + 1);

					}

				}

			}
		}
	}

	return b;


}



Eigen::VectorXf getB_mixture(Mat& src, Mat &dest, int **Index, Mat & mask)
{
	unsigned int unknownSize = dest.cols*dest.rows;
	Eigen::VectorXf b(unknownSize);

	// handle boundary cases

	int currentX_index;
	for (unsigned int i = 0; i < dest.rows; i++)
	{
		for (unsigned int j = 0; j < dest.cols; j++)
		{

			float current_value_dst = dest.ATF(i, j);
			currentX_index = Index[i][j];

			b(currentX_index) = 0;
			if (mask.ATU(i, j) == 0)
			{
				b(currentX_index) = current_value_dst;
			}
			else
			{

				float current_value = src.ATF(i, j);

				//i-1,j
				if ((i - 1) >= 0)
				{
					if (abs(current_value_dst - dest.ATF(i - 1, j )) > abs(current_value - src.ATF(i - 1, j)))
					{
						b(currentX_index) += current_value_dst - dest.ATF(i - 1 , j );

					}
					else
					{
						b(currentX_index) += current_value - src.ATF(i - 1, j);
					}



					//b(currentX_index) = b(currentX_index) + current_value - src.ATF(i - 1 - p.x, j - p.y);
					if (mask.ATU(i - 1, j) == 0)
					{
						b(currentX_index) = b(currentX_index) + dest.ATF(i - 1, j);

					}

				}

				//i+1,j
				if ((i + 1) <= (dest.rows - 1))
				{

					if (abs(current_value_dst - dest.ATF(i + 1 , j)) > abs(current_value - src.ATF(i + 1, j)))
					{
						b(currentX_index) += current_value_dst - dest.ATF(i + 1 , j );

					}
					else
					{
						b(currentX_index) += current_value - src.ATF(i + 1, j);
					}

					//b(currentX_index) = b(currentX_index) + current_value - src.ATF(i + 1 - p.x, j - p.y);
					if (mask.ATU(i + 1, j) == 0)
					{
						b(currentX_index) = b(currentX_index) + dest.ATF(i + 1, j);

					}

				}

				//i,j-1
				if ((j - 1) >= 0)
				{

					if (abs(current_value_dst - dest.ATF(i, j - 1 )) > abs(current_value - src.ATF(i, j - 1)))
					{
						b(currentX_index) += current_value_dst - dest.ATF(i , j - 1 );

					}
					else
					{
						b(currentX_index) += current_value - src.ATF(i, j - 1);
					}


					//b(currentX_index) = b(currentX_index) + current_value - src.ATF(i - p.x, j - 1 - p.y);
					if (mask.ATU(i, j - 1) == 0)
					{
					b(currentX_index) = b(currentX_index) + dest.ATF(i, j - 1);

					}

				}
				//i,j+1
				if ((j + 1) <= (dest.cols - 1))
				{
					if (abs(current_value_dst - dest.ATF(i, j + 1 )) > abs(current_value - src.ATF(i, j + 1)))
					{
						b(currentX_index) += current_value_dst - dest.ATF(i, j + 1 );

					}
					else
					{
						b(currentX_index) += current_value - src.ATF(i, j + 1);
					}

					//b(currentX_index) = b(currentX_index) + current_value - src.ATF(i - p.x, j + 1 - p.y);
					if (mask.ATU(i, j + 1) == 0)
					{
					b(currentX_index) = b(currentX_index) + dest.ATF(i, j + 1);

					}

				}

			}
		}
	}

	return b;


}


SpMat getA(Mat & Mask, int** Index, unsigned int unknownSize)
{
	vector<T> A_s;

	int currentX_index;
	for (unsigned int i = 0; i < Mask.rows; i++)
	{
		for (unsigned int j = 0; j < Mask.cols; j++)
		{
			currentX_index = Index[i][j];

			if (Mask.ATU(i, j) == 0)
			{
				A_s.push_back(T(currentX_index, currentX_index, 1));
			}
			else{
					{
						// define if the mask on the image edge
						int neighbour_count = 0;

						if (i - 1 >= 0)
						{
							neighbour_count++;
						}

						if ((i + 1) <= (Mask.rows - 1))
						{
							neighbour_count++;
						}

						if (j - 1 >=0)
						{
							neighbour_count++;
						}

						if ((j + 1) <= (Mask.cols - 1))
						{
							neighbour_count++;
						}

						A_s.push_back(T(currentX_index, currentX_index, neighbour_count));

						//i-1,j
						if (i - 1 >= 0){
							if (Mask.ATU(i - 1, j) == 255)
							{
								A_s.push_back(T(currentX_index, Index[i - 1][j], -1));

							}

						}



						//i+1,j
						if ((i + 1) <= (Mask.rows - 1))
						{

							if (Mask.ATU(i + 1, j) == 255)
							{
								A_s.push_back(T(currentX_index, Index[i + 1][j], -1));

							}
						}

						//i,j-1
						if (j >= 1)
						{

							if (Mask.ATU(i, j - 1) == 255)
							{
								A_s.push_back(T(currentX_index, Index[i][j - 1], -1));

							}
						}
						//i,j+1

						if ((j + 1) <= (Mask.cols - 1))
						{

							if (Mask.ATU(i, j + 1) == 255)
							{
								A_s.push_back(T(currentX_index, Index[i][j + 1], -1));

							}
						}
					}

			}


		}
	}


	SpMat A_sparse(unknownSize, unknownSize);
	A_sparse.setFromTriplets(A_s.begin(), A_s.end());
	return A_sparse;


}


