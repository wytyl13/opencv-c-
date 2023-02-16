/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-02-14 17:12:06
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-02-14 17:12:06
 * @Description: this file involved some general method for digital image processing
 * and some program processing tool, just like the statistic method in digital image processing.
 * some variable or class, struct we needed in the program. and some function that suitable
 * for all program. and so on.
 * then, we should consider the mean, variance, standard deviation, covariance and so on.
 * mean, variance, standard deviation is simple.
 * mean of one image show the mean gray value for this image.
 * variance or standard deviation of one image show the contrast of one image.
 * then, we will deep understand the application of other statistic method in digital image processing.
 * convariance.
 * convariance_xy = Σi=0_n[(x_i - μ_x)*(y_i - μ_y)] / (n - 1)
 * set or image x and y are positive correlation if convariance_xy is greater than zero.
 * or they are negative correlation or zero correlation.
 * but convariance can not measure the related degree. so we introduce the correlation coefficient.
 * the correlation coefficient is equal to convariance_xy / standard deviation. the range value of
 * correlation is from -1 to 1.
 * you can find based on the expression, the convariance_xx is a special form for the convariance_xy
 * convariance_xx == convariance = Σi=0_n[(x_i - μ_x)*(x_i - μ_x)] / (n - 1)
 * we have got the two dimension convariance, just like convariance_xy, but how to do calculate if 
 * we want to calculate convariance_xyz? then, we will use the convariance matrix. it is still calculating
 * the convariance, the convariance_xyz involved convariance_xx, convariance_xy, convariance_xz, 
 * convariance_yy, convariance_yx, convariance_yz, convariance_zz, convariance_zx, convariance_zy
 *     x   y   z
 * x  xx  xy   xz 
 * 
 * y  yx  yy   yz
 * 
 * z  zx  zy   zz
 * so this is a convariance matrix. convariance_xy == convariance_yx
 * so the convariance matrix is a symmetric matrix.
 * you can calculate them used official method in opencv.
 * 
 * then, we will consider the feature value and feature vector. 
 * assuming that A is a n order matrix. if A*x = lamda*x and x is a n dimension nonzero column vector. 
 * lamda is a real number. it means the column vector just scale not rotation. then we can named the lamda
 * is feature value of the n order matrix.
 * 
 * then, how to calculate the feature value and feature vector? it is a aimple problem in higher mathmatics.
 * you can calculate them based on the feature polynomial. you can find the resolve method in other region. 
 * we will consider the meaningful of feature value and feature vector at here.
 *  the first meaningful of A*x = lamda*x, x only the scaling under the effect of matrix A. the scale
 * is lamda. they show t the linear transformation.
 * the geometric meaning of feature value and feature vector is a space scaled based on the feature value scale rate
 * follow the direction of the feature vector. what is the space? it is a matrix, but it is generally
 * convariance matrix in the application of digital image processing.
 * 
 * the meaning of the feature vector of convariance matrix is to find the max projection direction that 
 * the variance in the direction of feature vector. you can image one two dimension coordinates figure.
 * you can find many direction, but you must can find a max projection variance in these direction.
 * what is the projection variance? you can image it is the coordinates mapping the direction you selected. 
 * of course, you can also select x axis and y axis as the direction. you can also select the feature vector
 * of the convariance matrix.
 * the max projection can remain large useful information. just like PCA algorithm, you will have to find 
 * the max projection direction to remain enough useful information. or you will get a badly result.
 * so the application of feature value and feature vector in digital image process is it can
 * find the most efficient direction that can reduce the amount of calculation, and it can remain much
 * useful information.
 *    
 * 4|      .  ... ..B
 *  |  . .. .. .
 * 1|A.. .....
 *  |
 *  |0______________8______
 * just like the case above, if you selected x axis as the vector direction, you will get the mapping range
 * from 0 to 8, but you will get the mapping range from 1 to 4 if you selected y axis as the vector direction.
 * of course, you can selected the feature vector of the convariance matrix as the vector direction. the feature
 * vector of convariance matrix must has the max projection, so it must the vector that similar to from point
 * A to point B. so you can also selected the vector used this method in PCA algorithm.
 * 
 * you can use the official method eigen to calculate feature value and feature vector.
 * bool eigen(InputArray src, OutputArray eigenValue, OutputArray eigenvectors = noArray());
 * 
***********************************************************************/
#ifndef _GENERAL_H
#define _GENERAL_H
#include <iostream>
#include <cstdarg>
#include <string.h>
#include <vector>
#include <map>
#include <dirent.h>
#include <sys/types.h>

#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/face/facerec.hpp>

#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

// notice, the dense head file must be the former than the eigen.hpp
// or you will get error. if you include dlib, the eigen will warn you.
// and the compiler speed will be low if you link the static library.
// and the run time will be speed if you use the static library.
// so we will annotation the code about eigen. and open it when we need to use it.
#if 0
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#endif
#include <Python.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace face;
// using namespace dlib;


#define FACEDETECTMODEL1 "C:/Users/weiyutao/opt/opencv/source/opencv-4.7.0/data/haarcascades/haarcascade_frontalface_alt2.xml"
#define EYEDETECTMODEL "C:/Users/weiyutao/opt/opencv/source/opencv-4.7.0/data/haarcascades/haarcascade_eye.xml"
#define FACEDETECTMODEL2 "C:/Users/weiyutao/opt/opencv/source/opencv-4.7.0/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
// of course, you can also use enum keywork to instead define. the difference between define adn enum is 
// the last is more flexible. just like define can just define the global variable, but the enum is not.
// the enum class is default started from 0, ++, if you want to make it start form the other number,
// you should define the first attribution FAST = anynumber. 
// you can define a enum class, and then use each attribution in your process. just like FAST is zero in this case.
// but how to define the enum class in head file? you can image enum as a struct or a class.
// it is a statement. so you can statement it in head file. and how to use it in cpp or c file?
// you can define the enum class first, just like FD fd; then call the attribution in this struct,
// you can also use FD::FAST direct.
typedef enum FeatureDetection
{
    FAST,
    ORB,
    SIFT,
    SURF,
    DEEPLEARNING 
} FD;

typedef enum SpatialFilterDevice
{
    SHARPEN,
    FUZZY
} SFD;

typedef enum AttributiionOfCoordinatesSet
{
    LENGTH,
    AREA,
    CENTER
} ACS;

typedef enum DlibFaceDetectAlgorithm
{
    HOG,
    MMOD
} DLIB;

class compareMap
{
public:
    bool operator()(int val1, int val2)
    {
        // descending order.
        return val1 > val2;
    }
};

string operator+(string &content, int number);
string& operator+=(string &content, int number);

void *calculateAttributionBasedOnFeaturePoints(vector<Point> &pts, int mode);

void sys_error(const char *str);
void elementOperation(Mat inputImage, Mat &outputImage);
void imshowMulti(string &str, vector<Mat> vectorImage);
void printMap(const map<int, Mat, compareMap> &mapBitPlane);
void printOneArrayPointer(const double *array, int length);
void printTwoArrayPointer(const double *array1, const double *array2, int lenght);
void printArrayListPointer(double *array, ...);
void thread_function(int i);
void freePointer(void *pointer);

void drawLines(Mat &inputImage, Point one, Point two);
void drawPolygon(Mat &inputImage, vector<Point> vectorPoints);
void screenShots(Mat inputImage, Mat &outputImage, Rect rect);
void cutImage(Mat inputImage, Mat &outputImage, vector<Point> vectorPoints);

#endif