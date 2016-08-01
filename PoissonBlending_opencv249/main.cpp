
#include "Header.h"

using namespace std;
using namespace cv;

#define ATU at<uchar>

int main(int argc, char* argv[])
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
//	cin >> num;
	cout << endl;
	long start_time = 0, end_time = 0;
	start_time = clock();

	string folder = "../img/";

	string source_path = folder + "s2.png";
	string dest_path = folder + "s1.png";
	string mask_path = folder + "mask5.png";

	Mat source = imread(source_path, IMREAD_COLOR);
	Mat dest = imread(dest_path, IMREAD_COLOR);
	Mat mask = imread(mask_path, IMREAD_COLOR);

	if (source.empty())
	{
		cout << "Could not load source image " << source_path << endl;
		return(-1);
	}
	if (dest.empty())
	{
		cout << "Could not load destination image " << dest_path << endl;
		return(-1);
	}
	if (mask.empty())
	{
		cout << "Could not load mask image " << mask_path << endl;
		return(-1);
	}


	if (num == 1)
	{
		long start_time = 0, end_time = 0;
		start_time = clock();

		Point p;

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

		p.x = leny / 2 + 105;  // x y are opposite from my code
		p.y = lenx / 2 + 200;

		Mat result;

		seamlessClone(source, dest, mask, p, result, 1);

		imshow("Output", result);

		imwrite(folder + "cloned2.png", result);
	}

	end_time = clock();
	cout << "Time cost: " << ((double)(end_time - start_time)) / CLOCKS_PER_SEC << " second" << endl;
	waitKey(0);

	return 0;
}
