/*
* cloning_demo.cpp
*
* Author:
* Siddharth Kherada <siddharthkherada27[at]gmail[dot]com>
*
* This tutorial demonstrates how to use OpenCV seamless cloning
* module without GUI.
*
* 1- Normal Cloning
* 2- Mixed Cloning
* 3- Monochrome Transfer
* 4- Color Change
* 5- Illumination change
* 6- Texture Flattening

* The program takes as input a source and a destination image (for 1-3 methods)
* and ouputs the cloned image.
*
* Download test images from opencv_extra folder @github.
*
*/

#define ATU at<uchar>

#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace cv;

int main()
{
	cout << endl;
	cout << "Cloning Module" << endl;
	cout << "---------------" << endl;
	cout << "Options: " << endl;
	cout << endl;
	cout << "1) Normal Cloning " << endl;
	cout << "2) Mixed Cloning " << endl;
	cout << "3) Monochrome Transfer " << endl;
	cout << "4) Local Color Change " << endl;
	cout << "5) Local Illumination Change " << endl;
	cout << "6) Texture Flattening " << endl;
	cout << endl;
	cout << "Press number 1-6 to choose from above techniques: ";
	int num = 1;
	cin >> num;
	cout << endl;
	long start_time = 0, end_time = 0;
	start_time = clock();

	if (num == 1)
	{
		long start_time = 0, end_time = 0;
		start_time = clock();

		Point p;
	//	p.x = destination.size().width / 2;
	//	p.y = destination.size().height / 2;

		string folder = "../img/";
		/*string original_path1 = folder + "face.png";
		string original_path2 = folder + "lisa.png";
		string original_path3 = folder + "face_mask.png";*/

		/*string original_path1 = folder + "src.jpg";
		string original_path2 = folder + "dst.jpg";
		string original_path3 = folder + "src_mask.jpg";*/

		string original_path1 = folder + "s2.png";
		string original_path2 = folder + "s1.png";
		string original_path3 = folder + "mask3.png";
		

		Mat source = imread(original_path1, IMREAD_COLOR);
		Mat destination = imread(original_path2, IMREAD_COLOR);
		Mat mask = imread(original_path3, IMREAD_COLOR);

		if (source.empty())
		{
			cout << "Could not load source image " << original_path1 << endl;
			exit(0);
		}
		if (destination.empty())
		{
			cout << "Could not load destination image " << original_path2 << endl;
			exit(0);
		}
		if (mask.empty())
		{
			cout << "Could not load mask image " << original_path3 << endl;
			exit(0);
		}

		Mat gray = Mat(mask.size(), CV_8UC1);

		if (mask.channels() == 3)
			cvtColor(mask, gray, COLOR_BGR2GRAY);

		else
			gray = mask;

		int minx = INT_MAX, miny = INT_MAX, maxx = INT_MIN, maxy = INT_MIN;
		int h = gray.size().height;
		int w = gray.size().width;

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

	//	cout << "( " << lenx / 2 << "," << leny / 2 << " )" << endl;
	//	cout << "( " << (minx + maxx) / 2 << "," << (miny + maxy) / 2 << " )" << endl;

		/*int minxd = p.y - lenx / 2;
		int maxxd = p.y + lenx / 2;
		int minyd = p.x - leny / 2;
		int maxyd = p.x + leny / 2;*/
		
	//	p.x = source.size().width / 2 ; // the position where in the dest image for the center of mask 
	//	p.y = source.size().height / 2 ;

		p.x = leny / 2 + 105;  // x y are opposite from my code
		p.y = lenx / 2 + 200;

	//	cout << "px=" << p.x << " py=" << p.y << endl;
		Mat result;
		

		seamlessClone(source, destination, mask, p, result, 1);
	    
		imshow("Output", result);
	
		imwrite(folder + "cloned.png", result);
	}
	else if (num == 2)
	{
		

		string folder = "../img/";
		/*string original_path1 = folder + "face.png";
		string original_path2 = folder + "lisa.png";
		string original_path3 = folder + "face_mask.png";*/
		string original_path1 = folder + "src.jpg";
		string original_path2 = folder + "dst.jpg";
		string original_path3 = folder + "src_mask.jpg";

		Mat source = imread(original_path1, IMREAD_COLOR);
		Mat destination = imread(original_path2, IMREAD_COLOR);
		Mat mask = imread(original_path3, IMREAD_COLOR);

		if (source.empty())
		{
			cout << "Could not load source image " << original_path1 << endl;
			exit(0);
		}
		if (destination.empty())
		{
			cout << "Could not load destination image " << original_path2 << endl;
			exit(0);
		}
		if (mask.empty())
		{
			cout << "Could not load mask image " << original_path3 << endl;
			exit(0);
		}

		Mat result;
		Point p;
		p.x = destination.size().width / 2 - 300;
		p.y = destination.size().height / 2 - 300;
	
		seamlessClone(source, destination, mask, p, result, 2);

		imshow("Output", result);
		imwrite(folder + "cloned.png", result);

		

	}
	else if (num == 3)
	{
		string folder = "cloning/Monochrome_Transfer/";
		string original_path1 = folder + "source1.png";
		string original_path2 = folder + "destination1.png";
		string original_path3 = folder + "mask.png";

		Mat source = imread(original_path1, IMREAD_COLOR);
		Mat destination = imread(original_path2, IMREAD_COLOR);
		Mat mask = imread(original_path3, IMREAD_COLOR);

		if (source.empty())
		{
			cout << "Could not load source image " << original_path1 << endl;
			exit(0);
		}
		if (destination.empty())
		{
			cout << "Could not load destination image " << original_path2 << endl;
			exit(0);
		}
		if (mask.empty())
		{
			cout << "Could not load mask image " << original_path3 << endl;
			exit(0);
		}

		Mat result;
		Point p;
		p.x = destination.size().width / 2;
		p.y = destination.size().height / 2;

		seamlessClone(source, destination, mask, p, result, 3);

		imshow("Output", result);
		imwrite(folder + "cloned.png", result);
	}
	else if (num == 4)
	{
		string folder = "cloning/Color_Change/";
		string original_path1 = folder + "source1.png";
		string original_path2 = folder + "mask.png";

		Mat source = imread(original_path1, IMREAD_COLOR);
		Mat mask = imread(original_path2, IMREAD_COLOR);

		if (source.empty())
		{
			cout << "Could not load source image " << original_path1 << endl;
			exit(0);
		}
		if (mask.empty())
		{
			cout << "Could not load mask image " << original_path2 << endl;
			exit(0);
		}

		Mat result;

		colorChange(source, mask, result, 1.5, .5, .5);

		imshow("Output", result);
		imwrite(folder + "cloned.png", result);
	}
	else if (num == 5)
	{
		string folder = "cloning/Illumination_Change/";
		string original_path1 = folder + "source1.png";
		string original_path2 = folder + "mask.png";

		Mat source = imread(original_path1, IMREAD_COLOR);
		Mat mask = imread(original_path2, IMREAD_COLOR);

		if (source.empty())
		{
			cout << "Could not load source image " << original_path1 << endl;
			exit(0);
		}
		if (mask.empty())
		{
			cout << "Could not load mask image " << original_path2 << endl;
			exit(0);
		}

		Mat result;

		illuminationChange(source, mask, result, 0.2f, 0.4f);

		imshow("Output", result);
		imwrite(folder + "cloned.png", result);
	}
	else if (num == 6)
	{
		string folder = "cloning/Texture_Flattening/";
		string original_path1 = folder + "source1.png";
		string original_path2 = folder + "mask.png";

		Mat source = imread(original_path1, IMREAD_COLOR);
		Mat mask = imread(original_path2, IMREAD_COLOR);

		if (source.empty())
		{
			cout << "Could not load source image " << original_path1 << endl;
			exit(0);
		}
		if (mask.empty())
		{
			cout << "Could not load mask image " << original_path2 << endl;
			exit(0);
		}

		Mat result;

		textureFlattening(source, mask, result, 30, 45, 3);

		imshow("Output", result);
		imwrite(folder + "cloned.png", result);
	}

	end_time = clock();
	cout << "Time cost: " << ((double)(end_time - start_time)) / CLOCKS_PER_SEC << " second" << endl;
	waitKey(0);
}
