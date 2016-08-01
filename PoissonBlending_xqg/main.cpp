
#include "Header_xqg.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	cout << endl;
	cout << "Cloning Module" << endl;
	cout << "---------------" << endl;
	cout << "Options: " << endl;
	cout << endl;
	cout << "1) Normal Cloning " << endl;
	cout << "2) Mixed Cloning " << endl;
	
	cout << endl;
	cout << "Press number 1-2 to choose from above techniques: ";
	int num = 1;
	cin >> num;
	cout << endl;

	long start_time = 0, end_time = 0;
	start_time = clock();

	Point p;
	//	p.x = 100;
	//	p.y = 100;
	

	string folder = "../img/";

	/*string original_path1 = folder + "face.png";
	string original_path2 = folder + "lisa.png";
	string original_path3 = folder + "face_mask4_3.png";
	p.x = 40;
	p.y = 40;*/

	string original_path1 = folder + "s2.png";
	string original_path2 = folder + "s1.png";
	string original_path3 = folder + "mask5.png";
	p.x = 200; // vertical
	p.y = 105; // horizontal

	//string original_path1 = folder + "car.png";
	//string original_path2 = folder + "street.png";
	//string original_path3 = folder + "car_mask.png";
	//p.x = 200; //destination.size().width / 2;
	//p.y = 130; 

	//string original_path1 = folder + "source1.png";
	//string original_path2 = folder + "sky.jpg";
	//string original_path3 = folder + "source1_mask.png";
	//p.x = 200; //destination.size().width / 2;
	//p.y = 600;

	/*string original_path1 = folder + "bear.jpg";
	string original_path2 = folder + "water.png";
	string original_path3 = folder + "bear_mask.bmp";
	p.x = 50;
	p.y = 20; */

	/*string original_path1 = folder + "shark.jpg";
	string original_path2 = folder + "beach.jpg";
	string original_path3 = folder + "shark_mask.png";
	p.x = 0;
	p.y = 0;*/

	/*string original_path1 = folder + "willuncle.jpg";
	string original_path2 = folder + "oranges.jpg";
	string original_path3 = folder + "manface_mask.png";
	p.x = 178;
	p.y = 165;*/

	/*string original_path1 = folder + "source1.png";
	string original_path2 = folder + "destination1.png";
	string original_path3 = folder + "source1_mask.png";
	p.x = 20;
	p.y = 300;*/

	/*string original_path1 = folder + "book.jpg";
	string original_path2 = folder + "wall.jpg";
	string original_path3 = folder + "book_mask.png";
	p.x = 50;
	p.y = 20; */

	/*string original_path1 = folder + "src.jpg";
	string original_path2 = folder + "dst.jpg";
	string original_path3 = folder + "src_mask.jpg";
	p.x = 100;
	p.y = 500;*/

	/*string original_path1 = folder + "src.jpg";
	string original_path2 = folder + "sky2.jpg";
	string original_path3 = folder + "src_mask.jpg";
	p.x = 100;
	p.y = 500;*/
	
	Mat source = imread(original_path1, IMREAD_COLOR);
	Mat destination = imread(original_path2, IMREAD_COLOR);
	Mat mask = imread(original_path3, IMREAD_COLOR);

	if (source.empty())
	{
		cout << "Could not load source image " << original_path1 << endl;
		return(-1);
	}
	if (destination.empty())
	{
		cout << "Could not load destination image " << original_path2 << endl;
		return(-1);
	}
	if (mask.empty())
	{
		cout << "Could not load mask image " << original_path3 << endl;
		return(-1);
	}

	Mat result;
//	p.x = destination.size().width / 2;
//	p.y = destination.size().height / 2;
	if (num == 1)
	{
		seamlessClone_xqg(source, destination, mask, p, result, 1);
	//	seamlessClone_xqg(source_yuv, destination_yuv, mask_yuv, p, result, 1);
	}
	
	else if (num == 2)
	{
		seamlessClone_xqg(source, destination, mask, p, result, 2);
	}
	else
	{
		cout << "Input wrong number, try again." << endl;
		return (-1);
	}

	end_time = clock();
	cout << "Time cost: " << ((double)(end_time - start_time)) / CLOCKS_PER_SEC << " second" << endl;
	waitKey(0);
//	system("pause");
	return 0;
}

