#ifndef _GENERAL_H
#define _GENERAL_H
#include <iostream>
#include <string.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;

void sys_error(const char *str);
void elementOperation(Mat inputImage, Mat &outputImage);
void imshowMulti(string &str, vector<Mat> vectorImage);


#endif