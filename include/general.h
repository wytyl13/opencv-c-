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
 * some basic matrix operation.
 * the official method in opencv, meanStdDev(inputImage, mean, std)
 * the inverse matrix, inv(inputImage)
 * the transpose matrix, inputImage.t()
 * the inner product of two matrix, inputImage.mul(inputImage2)
 * the @ of two matrix, inputImage * inputImage2.
 * the number of nonzero in one image. countNonZero(inputImage).
 * the min, max value in one matrix. minMaxLoc(inputImage, &min, &max, 0, 0)
 * the feature value and feature vector of one matrix. eigen(inputImage, featureValue, featureVector).
 * then, how to calculate the rank, determinant value, adjoint matrix of one matrix.
 * the adjoint matrix is equal to the product of the determinant value and the inverse of the matrix.
 * inverse(A) = (1/|A|) * A*, so A* = inverse(A) * |A|.
 * we have known how to calculate the inverse of matrix, so we just need to know how to calculate the 
 * determinant value of one matrix.
 * the official method about the determinant value of one matrix is double value = determinant(inputImage);
 * notice, the determinant function in opencv has many limit conditions, just like
 * Assertion failed (mat.rows == mat.cols && (type == CV_32F || type == CV_64F)) in determinant
 * 
 * then, we will consider how to calculate the rank of one matrix.
 * the rank of one matrix is equal to the numbers of vectors of the maxinum linearly independent group in
 * the group of the column vectors of one matrix.
 * 
 * notice, the matrix could not be separable if the determinant value of one matrix is greated than 1 or less
 * than 1.
 * the concept of rank is the min order son type of the matrix. we can conclude the min rank for one matrix
 * is one based on the feature of one matrix. if the matrix has nonzero value, the min rank of this matrix
 * must be 1, if the matrix is zero marix, the rank of matrix must be zero.
 * so we can use the rank to judge whether the matrix can be separable.

 *  
 * you should use %Lf if you want to format the long double. you should add .n behind % symbol
 * if you want to remain n bit decimal. 
 * 
 * how to calculate the rank of one matrix?
 * you can use eigen library. svd.rank(). you can also define the rank function by yourself.
 * the concept of calculating rank has two method.
 * one is gaussian elimination, another is singular matrix.
 * 
 * Eigen::MatrixXd temp;
 * cv::cv2eigen(FUZZYKERNEL, temp);
 * Eigen::JacobiSVD<Eigen::MatrixXd> svd(temp);
 * int rank = svd.rank();
 * 
 * we can conclude that the determinant value of one m order matrix must be zero if the rank of one matrix is 1.
 * so we can get one simple method to separate the m order matrix. we have described it in separateKernel function
 * in genearl.cpp file.
 * 
***********************************************************************/
#ifndef _GENERAL_H
#define _GENERAL_H

// notice, namespace cv is complicted with dlib, namespace Eigen is complicted with dlib.
// so you should open or close the macro in the suitable time.
#define ISOPENDLIB 0
#define ISOPENEIGEN 1
#define ISOPENHISTOGRAMTRANSFORM 0
#define ISOPENFACEAPPLICATION 0
#define ISOPENSPATIALFILTER 1
#define SOMEENUM 1
#define SOMEKERNEL 1
#define SOMEXMLFILES 1
#define ISOPENTEMPLATEFUNCTION 1

#include <iostream>
#include <cstdarg>
#include <string.h>
#include <vector>
#include <map>
#include <dirent.h>
#include <sys/types.h>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>

#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/face/facerec.hpp>

#if 0
// notice, if you can not find the head file or other executable file, you can find
// it in contrib what is the additional library for opencv. of course, you can include its
// include file used cmakeLists.txt, we have add the include path OpenCV_Contrib = 
// C:/Users/weiyutao/opt/opencv/source/opencv_contrib-4.7.0/modules, you can include the head file in it.
#endif

// include the head file about the additional library for opencv. there are some head file
// where is in opencv former, but last move them into the additional library. the opencv_creatsamples
// is the important library what we can use it train our classifier. it is in opencv source former.
// but new version opencv has moved it into the additional library. so we should find it in contrib.
// but we have failed to find the createsamples, because it has been disable in new opencv version.
#include <face/include/opencv2/face.hpp>
#include <freetype/include/opencv2/freetype.hpp>

#if ISOPENDLIB
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/object_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn/core.h>
#include <dlib/dnn/layers.h>
#endif
// notice, the dense head file must be the former than the eigen.hpp
// or you will get error. if you include dlib, the eigen will warn you.
// and the compiler speed will be low if you link the static library.
// and the run time will be speed if you use the static library.
// so we will annotation the code about eigen. and open it when we need to use it.
#if ISOPENEIGEN
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace Eigen;
#endif

#include <Python.h>

using namespace std;
using namespace cv;
using namespace face;
// using namespace dlib;

#if SOMEXMLFILES
// these are model that has trained successful as follow.
// then, we can describe the file as follow. the former three xml file is characteristics of
// cascade classifier for opencv. it is the traditional face detected based on the HAAR.
// the fourth dat file is mmod model what trained based on the neural network. it is more accuracy but low efficiency.
// the fifth dat file is 68 feature detected what dedicated to face in dlib. so you should use it in face not the picture.
// the sixth dat file is 128 dimension face feature detected based on the residual neural network.
// of course, you can also use the default face detector HOG what dlib has provide. it is also the characteristic of
// cascade classifier. 
#define OPENCVHAARFACEDETECT "C:/Users/weiyutao/opt/opencv/source/opencv-4.7.0/data/haarcascades/haarcascade_frontalface_alt2.xml"
#define OPENCVHAAREYEDETECT "C:/Users/weiyutao/opt/opencv/source/opencv-4.7.0/data/haarcascades/haarcascade_eye.xml"
#define OPENCVHAARFACEDETECT_EXTRA "C:/Users/weiyutao/opt/opencv/source/opencv-4.7.0/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
#define DLIBMMODMODELFACEDETECT "../../resources/model/mmod_human_face_detector.dat"
#define DLIBFACEFEATUREDETECT "../../resources/model/shape_predictor_68_face_landmarks.dat"
#define DLIBRESNETMODEL "../../resources/model/dlib_face_recognition_resnet_model_v1.dat"
#endif

#if SOMEKERNEL
// define some kernel, stored them used Mat. notice, Mat_ is one defined method used template,
// it can return one Mat class instance. just like image sharpening, it belong to the edge enhance.
// we have failed to test these shapen kernels used the spatialFilter function what we defined, but 
// they are successful to test used official function filter2D, then, we will define the other kernel.
// you can define any type Mat as the kernel, we will transformed them to double in super function.

// the exchange domain filter. sharpen is high-pass filter. high-pass and low-pass is dedicated to the frequency domain.
// notice, there are four sharpen kernel, they can also be named as laplacian.
// and you should notice, the function of laplacian is similar to the second derivative of the gray value.
// you should do the convolution operation used these four laplacian, and you should based on the expression as follow.
// only this, you can get the efficient of sharpening.
// g(x, y) = f(x, y) + c * (the result of convolution used the laplacian), c = the former laplacian : -1 ? 1
// you can find the rule that the sum of the each element in laplacian is zero.
#define SHARPENKERNEL_ (Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0)
#define SHARPENKERNEL__ (Mat_<int>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1)
#define SHARPENKERNEL___ (Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0)
#define SHARPENKERNEL____ (Mat_<int>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1)
#define GAUSSIANKERNEL71 getGaussianKernel_(7, 1)
#define GAUSSIANKERNEL132 getGaussianKernel_(13, 2)
#define GAUSSIANKERNEL193 getGaussianKernel_(19, 3)

// the spatial domain filter.
// you can find the the smooth is light fuzzy. the efficient about them is general similar but not equal.
#define FUZZYKERNEL (Mat::ones(5, 5, CV_32F) / (float)(25))
#define SMOOTHKERNELCASSETTE (Mat::ones(3, 3, CV_32F) / (float)(9))
#define SMMOTHKERNELGAUSSIAN ((Mat_<double>(3, 3) << 0.3679, 0.6065, 0.3679, 0.6065, 1.0000, 0.6065, 0.3679, 0.6065, 0.3679) / 4.8976)
#define DENOISINGkERNELGAUSSIAN ((Mat_<double>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16)

#endif

#if ISOPENDLIB
// define the template type based on the resnet in dlib.
// notice the namespace if you have not using the namespace.
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;
 
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;
 
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
 
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
 
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
	dlib::input_rgb_image_sized<150>
	>>>>>>>>>>>>;
#endif



#if SOMEENUM
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
    MMOD,
    SHAPE68
} DLIB;

typedef enum degrees
{
    NINTY,
    ONEEIGHTZERO
} DEGREES;

typedef enum spatialFilter
{
    CORRELATION,
    CONVOLUTION
} SPATIALFILTER;


typedef enum operators
{
    ADD,
    SUB,
    MULTI,
    DIVIDE
} OPERATORS;


class compareMap
{
public:
    bool operator()(int val1, int val2)
    {
        // descending order.
        return val1 > val2;
    }
};

int extractNum(string &ss, char *ch);
bool compareVectorString(std::string str1, std::string str2);
#endif


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

#if 1
// define a function used template. you must implement the template class or
// function in head file. because the template<> just a modified type. so you should
// write your class or function behind the template. of course you can implement many function
// but it is not complex. just like the defined about resnet, you can define the net used
// template. and you can simple your function or class used template.
template<class T>
void printVector(vector<T> vectorAny)
{
    for (vector<std::string>::iterator it = vectorAny.begin(); it != vectorAny.end(); it++)
    {
        cout << *it << endl;
    }
}
#endif

void drawLines(Mat &inputImage, Point one, Point two);
void drawPolygon(Mat &inputImage, vector<Point> vectorPoints);
void screenShots(Mat inputImage, Mat &outputImage, Rect rect);
void cutImage(Mat inputImage, Mat &outputImage, vector<Point> vectorPoints);

void getImageFileFromDir(const string dir, std::vector<cv::String> &imageNames, std::vector<cv::String> &imagePaths, int &countImage);
void getAllFileFromDirAndCreatTrainData(const string directoryPath, vector<string> &imagePath,\
     const string txtPath, int &count);

#if ISOPENTEMPLATEFUNCTION
// define some generally used template function.
template<class T>    
void rotationVector(vector<vector<T>> &matrix)
{
    int n = matrix.size();
    if (n == 0)
    {
        return;
    }
    int r = (n >> 1) - 1;
    int c = (n - 1) >> 1;
    for (int i = r; i >= 0; --i)
    {
        for (int j = c; j >= 0; --j)
        {
            swap(matrix[i][j], matrix[j][n - i - 1]);
            swap(matrix[i][j], matrix[n - i - 1][n - j - 1]);
            swap(matrix[i][j], matrix[n - j - 1][i]);
        }
    }
}

#endif



void rotationMatVector(Mat &inputImage, int degrees);
void rotationMat90(Mat &inputImage);
void rotationMat(Mat &inputImage, int degrees);
int getRankFromMat(Mat &inputImage);
void separateKernel(Mat &w, Mat &w1, Mat &w2);

Mat getGaussianKernel_(const int size, const double sigma);
#endif