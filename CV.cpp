#include <iostream>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/features2d.hpp>

using namespace cv;
using namespace std;

struct CallbackParam
{
	Mat frame;
	Point pt1, pt2;
	Rect roi;
	bool drag;
	bool updated;
};

int count_mouse_click = 0;
double pointX[4], pointY[4];
double newpointX[4], newpointY[4];
double pre_pointX[4], pre_pointY[4];
double pre_newpointX[4], pre_newpointY[4];
double width;
double height;
//roi������ �ش�Ǵ� �� ���� ������ ����
CvPoint2D32f roi_array[10];


int caculate_start = 0;
Mat img_gray;
Mat homo(Mat & name);
int number_of_frame = 0;
Mat homo_h;

void CallBackFunc(int, int, int, int, void*);
float BilinearInterpolation(float, float, float, float, float, float, float, float, float, float);
void yellow_detect(Mat&, Mat&);
void hist_equalization(Mat&, Mat&);
void line_detect(Mat&, Mat&, Mat&);
void draw_contours(Mat&, Mat&);
void circle(Mat&, Mat&);
void hough_circle(Mat&, Mat&, Mat&);
Mat homo(Mat &);


int main(void)
{
	//������ �ε�
	//������ KakaoTalk_Video_20190616_2138_21_605
	//���׶�� KakaoTalk_Video_20190622_0915_01_769
	//���� KakaoTalk_Video_20190622_0914_42_294
	// https://webnautes.tistory.com/818
	Mat image_1;
	//// ����� ������ ������ �������ݴϴ�.
	VideoCapture cap("KakaoTalk_Video_20190622_0915_01_769.mp4");
	if (!cap.isOpened()) {
		cerr << "���� - ī�޶� �� �� �����ϴ�.\n";
		return -1;
	}

	while (1)
	{
		cap.read(image_1);
		if (image_1.empty()) {
			cerr << "�� �����Դϴ�.\n";
			break;
		}

		imshow("Color", image_1);
		if (waitKey(25) >= 0)
			break;

		Mat face = imread("LEE.jpg");
		resize(face, face, Size(), 0.1, 0.1);
		imshow("������ 21611663", face);
		////�̹��� �ε�
		//Mat image_1 = imread("homograph.jpg");
		////resize(image_1, image_1, Size(), 0.3, 0.3);
		//imshow("image", image_1);
		//waitKey();

		//�׷��̽����� �̹����� ��ȯ  
		cvtColor(image_1, img_gray, COLOR_BGR2GRAY);
		//resize(img_gray, img_gray, Size(), 1.5, 1.5);
		//������ ����  
		namedWindow("gray image", WINDOW_AUTOSIZE);
		//namedWindow("gray image", 0);
		//namedWindow("result image", WINDOW_AUTOSIZE);
		//�����쿡 ���  

		imshow("gray image", img_gray);

		////waitKey();
		//setMouseCallback("gray image", CallBackFunc, NULL);

		//cout << "���� �� - ������ �� - ���� �Ʒ�, ������ �Ʒ� ������ Ŭ�����ּ���" << endl;
		//waitKey();

		//����� �κ� ���� 
		//src ����: color
		//dst ���: ����� mask
		Mat mask = Mat::ones(image_1.rows, image_1.cols, image_1.type());
		yellow_detect(image_1, mask);

		//histogram matching
		//src ����: mask�� �ص� �ǰ�, ��������(image_1)�� �ص� ��
		//dst ���: gray ����, 3ä�� ������ �ƿ� ����
		Mat hist;
		hist_equalization(mask, hist);

		//hough line
		//hough line ���ַ��� canny edge ������
		//src ����: gray ����
		//dst ���: gray edge�� hough line �׷��� Mat,canny�� gray
		Mat hough_line, canny_edge;
		cvtColor(image_1, hist, CV_RGB2GRAY);
		line_detect(hist, hough_line, canny_edge);

		//homography�� ������ ��� ���� �� ������ �޾ƿ���
		//mask�� ���� ���, ������ ���, ���� �ϴ�, ������ �ϴ�

		//four_corners(mask);
		//Mat pointsrc, pointdst;
		//Mat transformMatrix = getPerspectiveTransform(pointX, pointY);
		//warpPerspective(mask, mask, transformMatrix, Size(width, height));
		//https://miatistory.tistory.com/5

		//findcontours
		//src ����: canny edge �� ����
		//dst ���: CV_8U����
		//Mat drawing = Mat::zeros(canny_edge.rows, canny_edge.cols, canny_edge.type());
		Mat drawing(canny_edge.size(), CV_8U);
		draw_contours(canny_edge, drawing);

		/*Mat circled;
		circle(drawing, drawing,circled);
		*/
		Mat circled;
		//circle(drawing, circled);

		hough_circle(drawing, image_1, circled);
		//http://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=220592858479&redirect=Dlog&widgetTypeCall=true

	}
	return 0;
}
//Ⱦ�ܺ��� �������
//Ⱦ�ܺ��� ����
//Ⱦ�ܺ��� ���� ������ �������� ����

//������׷� ��Ȱȭ
void hist_equalization(Mat& src, Mat& dst)
{
	//color input, color output 
	cvtColor(src, src, CV_BGR2GRAY);
	equalizeHist(src, dst);
	imshow("before", src);
	imshow("after", dst);
	//waitKey();

}
//����� �κ� ���� 
void yellow_detect(Mat& src, Mat& dst)
{
	Mat HSV;
	cvtColor(src, HSV, CV_BGR2HSV);
	vector<Mat> channels;
	split(HSV, channels);
	//cvInRangeS(&image, cvScalar(20, 100, 100), cvScalar(30, 255, 255), &mask);
	//https://opencvlib.weebly.com/cvinranges.html
	float H, S, V;

	for (int i = 0; i < HSV.rows; i++)
	{
		for (int j = 0; j < HSV.cols; j++)
		{
			H = HSV.at<Vec3b>(i, j)[0];
			S = HSV.at<Vec3b>(i, j)[1];
			V = HSV.at<Vec3b>(i, j)[2];

			if ((H > 10 && H < 30) && (S > 100 && S < 255) && (V > 100 && V < 255))
			{
				//������� �κи� mask�� �Ҵ�
				dst.at<Vec3b>(i, j) = HSV.at<Vec3b>(i, j);
			}
		}
	}
	imshow("mask", dst);//3ä��(�����) ǥ��
	//waitKey();
}

void line_detect(Mat& src, Mat& dst, Mat& canny)
{
	//dst=hough_line
	Mat gray;
	GaussianBlur(src, gray, Size(3, 3), 1.3);
	//imshow("GaussianBlured", gray); 	waitKey();
	Canny(gray, canny, 100, 200, 3);
	imshow("canny_edged", canny);// waitKey();
	cvtColor(canny, dst, CV_GRAY2RGB);//color �ȹٲ��ָ� error
	//RGB�� �ٲ㵵 �ᱹ gray��
	vector<Vec2f> lines;
	HoughLines(canny, lines, 1, CV_PI / 180, 200, 0, 0);//th�� ��������� �� th�� �������� ���� ���� �����

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}

	imshow("detected lines", dst);
	//waitKey();

}

void draw_contours(Mat& src, Mat& dst)
{
	////0616
	//cvtColor(src, src, CV_RGB2GRAY);//findcontours �Լ� ��밡������
	src.convertTo(src, CV_8UC1);
	//imshow("edge", src); 	waitKey();
	//Mat�� �ڷ�����  CV_16U�̰� findContours�Լ��� CV_8UC1���� ����Ѵ�.
	vector<vector<Point>> contours;
	findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// https://boysboy3.tistory.com/60

	drawContours(dst, contours, -1, cvScalar(0), 2);
	//imshow("Contours", dst);
	//waitKey();

}

//�� ����
void circle(Mat&src, Mat& dst)
{
	/// Detector parameters
	int blockSize = 10;
	int apertureSize = 10;
	double k = 0.04;

	SiftFeatureDetector sift;
	//sift = Feature2D;
	vector<KeyPoint> kps_db;
	sift.detect(src, kps_db);
	drawKeypoints(src, kps_db, dst, Scalar::all(-1), 4);
	imshow("result/21611663/������", dst);
	//waitKey(0);
}

void hough_circle(Mat&src, Mat& ref, Mat& dst)
{
	vector<Vec3f> circles;
	HoughCircles(src, circles, CV_HOUGH_GRADIENT, 2, src.rows / 20, 200, 40, 7, 22);
	//HoughCircles(src, circles, CV_HOUGH_GRADIENT,2, 100 );
	//����� �� ǥ��
	Mat circle_image = ref.clone();
	char zBuffer[35];
	if (circles.size() > 50)
	{//go,stop ���
		//copy the text to the "zbuffer"
		_snprintf_s(zBuffer, 35, "STOP");

	}
	else
	{
		_snprintf_s(zBuffer, 35, "GO");
	}
	//put the text in the "zbuffer" to the "dst" image
	putText(circle_image, zBuffer, Point(100, 100), CV_FONT_HERSHEY_PLAIN, 3, Scalar(255, 200, 0), 4);
	for (size_t i = 0; i < circles.size(); i++)
	{

		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		cout << radius << endl;
		//draw the circle center
		circle(circle_image, center, 1, Scalar(0, 255, 0), -1, 0, 0);
		//draw the circle outline
		circle(circle_image, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}
	_snprintf_s(zBuffer, 35, "21611663 LEEJUHYUNG");
	putText(circle_image, zBuffer, Point(250, 100), CV_FONT_HERSHEY_PLAIN, 2, Scalar(255, 200, 0), 4);

	namedWindow("Circle Detection", WINDOW_AUTOSIZE);
	imshow("Circle Detection", circle_image);
	//waitKey();
}


float BilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y)
{
	float x2x1, y2y1, x2x, y2y, yy1, xx1;
	x2x1 = x2 - x1;
	y2y1 = y2 - y1;
	x2x = x2 - x;
	y2y = y2 - y;
	yy1 = y - y1;
	xx1 = x - x1;
	return 1.0 / (x2x1 * y2y1) * (
		q11 * x2x * y2y +
		q21 * xx1 * y2y +
		q12 * x2x * yy1 +
		q22 * xx1 * yy1
		);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << count_mouse_click << "��° ���� ���콺 ��ư Ŭ��.. ��ǥ = (" << x << ", " << y << ")" << endl;
		pre_pointX[count_mouse_click] = x;
		pre_pointY[count_mouse_click] = y;

		count_mouse_click++;
	}

	if (count_mouse_click == 4 && caculate_start == 0)
	{
		caculate_start = 1;

		cout << "#######################################################" << endl;
		cout << "H��꿡 �ʿ��� 4���� ���� ��� Ŭ���߽��ϴ�." << endl << endl;


		width = ((pre_pointX[1] - pre_pointX[0]) + (pre_pointX[3] - pre_pointX[2]))*0.5;
		height = ((pre_pointY[2] - pre_pointY[0]) + (pre_pointY[3] - pre_pointY[1]))*0.5;

		pre_newpointX[0] = pre_pointX[3] - width;
		pre_newpointX[1] = pre_pointX[3];
		pre_newpointX[2] = pre_newpointX[0];
		pre_newpointX[3] = pre_newpointX[1];

		pre_newpointY[0] = pre_pointY[3] - height;
		pre_newpointY[1] = pre_newpointY[0];
		pre_newpointY[2] = pre_pointY[3];
		pre_newpointY[3] = pre_newpointY[2];

		for (int i = 0; i < 4; i++)
			cout << pre_newpointX[i] << " " << pre_newpointY[i] << endl;
		homo(img_gray);
	}
}

Mat homo(Mat & name)
{
	//inverse mapping
	Mat img_result = Mat::zeros(name.size(), CV_8UC1);

	rectangle(img_result, Point(pre_newpointX[0], pre_newpointY[0]), Point(pre_newpointX[3], pre_newpointY[3]), Scalar(255), 1);
	//imshow("img_gray image2", img_result);


	vector<Point2f> pts_src;
	vector<Point2f> pts_dst;

	for (int i = 0; i < 4; i++) {
		pts_src.push_back(Point2f(pre_pointX[i], pre_pointY[i]));
		pts_dst.push_back(Point2f(pre_newpointX[i], pre_newpointY[i]));
	}
	//Mat h222 = findHomography(pts_src, pts_dst);

	//cout << pts_src << endl;
	//cout << pts_dst << endl;
	//cout << h222 << endl;

	//nomalized DLT �˰��� ����.. 1/2  ����
	//���⿡���� ������ ��� �̹����� ���� ũ���ΰ����.

	int image_width = name.cols;
	int image_height = name.rows;

	Mat T_norm_old = Mat::zeros(3, 3, CV_64FC1);
	T_norm_old.at<double>(0, 0) = image_width + image_height;
	T_norm_old.at<double>(1, 1) = image_width + image_height;
	T_norm_old.at<double>(0, 2) = image_width * 0.5;
	T_norm_old.at<double>(1, 2) = image_height * 0.5;
	T_norm_old.at<double>(2, 2) = 1;

	Mat T_norm_new = Mat::zeros(3, 3, CV_64FC1);
	T_norm_new.at<double>(0, 0) = image_width + image_height;
	T_norm_new.at<double>(1, 1) = image_width + image_height;
	T_norm_new.at<double>(0, 2) = image_width * 0.5;
	T_norm_new.at<double>(1, 2) = image_height * 0.5;
	T_norm_new.at<double>(2, 2) = 1;


	for (int i = 0; i < 4; i++)
	{
		pointX[i] = (image_width + image_height)*pre_pointX[i] + image_width * 0.5;
		pointY[i] = (image_width + image_height)*pre_pointY[i] + image_height * 0.5;

		newpointX[i] = (image_width + image_height)*pre_newpointX[i] + image_width * 0.5;
		newpointY[i] = (image_width + image_height)*pre_newpointY[i] + image_height * 0.5;
	}
	///////////////////nomalized DLT �˰��� ����.. 1/2 ��

	double data[8][9] = {
		{ -1 * pointX[0], -1 * pointY[0], -1, 0, 0, 0, pointX[0] * newpointX[0], pointY[0] * newpointX[0], newpointX[0] },
		{ 0, 0, 0, -1 * pointX[0], -1 * pointY[0], -1, pointX[0] * newpointY[0], pointY[0] * newpointY[0], newpointY[0] },
		{ -1 * pointX[1], -1 * pointY[1], -1, 0, 0, 0,pointX[1] * newpointX[1], pointY[1] * newpointX[1], newpointX[1] },
		{ 0, 0, 0, -1 * pointX[1], -1 * pointY[1], -1,pointX[1] * newpointY[1], pointY[1] * newpointY[1], newpointY[1] },
		{ -1 * pointX[2], -1 * pointY[2], -1, 0, 0, 0,pointX[2] * newpointX[2], pointY[2] * newpointX[2], newpointX[2] },
		{ 0, 0, 0, -1 * pointX[2], -1 * pointY[2], -1,pointX[2] * newpointY[2], pointY[2] * newpointY[2], newpointY[2] },
		{ -1 * pointX[3], -1 * pointY[3], -1, 0, 0, 0,pointX[3] * newpointX[3], pointY[3] * newpointX[3], newpointX[3] },
		{ 0, 0, 0, -1 * pointX[3], -1 * pointY[3], -1,pointX[3] * newpointY[3], pointY[3] * newpointY[3], newpointY[3] },
	};
	Mat A(8, 9, CV_64FC1, data);
	cout << "Matrix A" << endl;
	cout << A << endl;

	Mat d, u, vt, v;
	SVD::compute(A, d, u, vt, SVD::FULL_UV);
	transpose(vt, v);

	//cout << "Matrix V" << endl;
	//cout << v << endl;

	Mat h(3, 3, CV_64FC1);

	//������ �÷����� H�� ���Ѵ�. 
	int lrow = 0;
	int lcols = v.cols - 1;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			h.at<double>(i, j) = v.at<double>(lrow, lcols);
			lrow++;
		}
	}

	//h_33�� 1�� �����.
	double dw = h.at<double>(2, 2);
	homo_h = h / dw;



	//nomalized DLT �˰��� ����.. 2/2 ����
	Mat T_norm_new_invert = T_norm_new.inv();
	homo_h = T_norm_new_invert * homo_h*T_norm_old;
	/////////////nomalized DLT �˰��� ����.. 2/2  ��

	for (int y = 0; y < img_result.rows; y++)
	{
		for (int x = 0; x < img_result.cols; x++)
		{
			double data[3] = { x, y,1 };
			Mat oldpoint(3, 1, CV_64FC1);
			Mat newpoint(3, 1, CV_64FC1, data);

			Mat h2 = homo_h.inv();
			oldpoint = h2 * newpoint;

			int oldX, oldY;

			oldX = (int)((oldpoint.at<double>(0, 0)) / (oldpoint.at<double>(2, 0)) + 0.5);
			oldY = (int)((oldpoint.at<double>(1, 0)) / (oldpoint.at<double>(2, 0)) + 0.5);


			if ((oldX >= 0 && oldY >= 0) && (oldX < img_result.cols && oldY < img_result.rows))
				img_result.at<uchar>(y, x) = name.at<uchar>(oldY, oldX);
		}
	}

	//������ ����
	Mat img_result2 = Mat::zeros(name.size(), CV_8UC1);

	for (int y = 1; y < img_result.rows - 1; y++)
	{
		for (int x = 1; x < img_result.cols - 1; x++)
		{
			int q11 = img_result.at<uchar>(y - 1, x - 1);
			int q12 = img_result.at<uchar>(y + 1, x - 1);
			int q21 = img_result.at<uchar>(y + 1, x + 1);
			int q22 = img_result.at<uchar>(y - 1, x + 1);

			if (img_result.at<uchar>(y, x) == 0)
			{
				int p = BilinearInterpolation(q11, q12, q21, q22, x - 1, x + 1, y - 1, y + 1, x, y);
				if (p > 255) p = 255;
				if (p < 0)  p = 0;

				img_result2.at<uchar>(y, x) = p;
			}
			else img_result2.at<uchar>(y, x) = img_result.at<uchar>(y, x);

		}
	}

	Rect rect(pre_newpointX[2], pre_newpointY[0], width, height);
	Mat subImage = img_result2(rect);

	//Mat resized_subImage;
	//resize(subImage, resized_subImage, Size(60, 552), 0, 0, CV_INTER_LINEAR);
	//��µǴ� top view�� ���μ��� ���� ����
	imshow("my result image2", subImage); waitKey();

	return subImage;


	//imshow("my result image2", resized_subImage); waitKey();
	//Size size(600, 600);
	//Mat im_dst = Mat::zeros(size, CV_8UC1);
	//warpPerspective(img_gray, im_dst, h222, size);

	//imshow("opencv result image3", im_dst);
}
