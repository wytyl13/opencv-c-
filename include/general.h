#ifndef _GENERAL_H
#define _GENERAL_H
#include <iostream>
#include <string.h>
#include <vector>
#include <map>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
// notice, the dense head file must be the former than the eigen.hpp
// or you will get error.
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace cv;
using namespace std;
using namespace Eigen;

class compareMap
{
public:
    bool operator()(int val1, int val2)
    {
        // descending order.
        return val1 > val2;
    }
};


void sys_error(const char *str);
void elementOperation(Mat inputImage, Mat &outputImage);
void imshowMulti(string &str, vector<Mat> vectorImage);
void printMap(const map<int, Mat, compareMap> &mapBitPlane);
void printArray(float *array);

void drawLines(Mat &inputImage, Point one, Point two);
void drawPolygon(Mat &inputImage, vector<Point> vectorPoints);
void screenShots(Mat inputImage, Mat &outputImage, Rect rect);
void cutImage(Mat inputImage, Mat &outputImage, vector<Point> vectorPoints);
#endif