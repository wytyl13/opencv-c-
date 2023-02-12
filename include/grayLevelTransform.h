#ifndef _GRAYLEVELTRANSFORM_H
#define _GRAYLEVELTRANSFORM_H
#include "general.h"
void contrastEnhance(Mat inputImage, Mat &outputImage);
void reverseTransform(Mat inputImage, Mat &outputImage);
void logarithmicAndLinearScaling(Mat inputImage, Mat &outputImage);
void linearScaling(Mat inputImage, Mat &outputImage);
void linearScalingBaseTwoPoint(Mat inputImage, Mat &outputImage);
void gamaTransformAndLinearScaling(Mat inputImage, Mat &outputImage, double c, double gama);
void grayLayeredBasedPoints(Mat inputImage, Mat &outputImage, vector<Point> vectorPoints, int mode);
void grayLayeredBasedValue(Mat inputImage, Mat &outputImage);
void grayLayeredBasedBitPlane(Mat inputImage, Mat &outputImage, int plane);
void reconstructImageBasedBitPlane(Mat &outputImage, const map<int, Mat, compareMap> &mapBitPlanes);
double* getDistribution(const Mat &inputImage);
void getHistogramMatBasedOnDistribution(double *distribution, Mat &histogramMat);
void getHistogramMatBasedOnInputImage(const Mat &inputImage, Mat &histogramMat);
double* getEqualizationDistribution(const Mat &inputImage);
void histogramEqualizeTransformation(const Mat &inputImage, Mat &outputImage);
double* getCumulativeHistogram(const Mat &inputImage);
void histogramMatchingTransformation(const Mat &inputImage, const Mat &objectImage, Mat &outputImage);
void histogramTransformationLocal(const Mat &inputImage, Mat &outputImage, int sideLength);
void histogramTransformationLocalThread(const Mat &inputImage, Mat &outputImage, int sideLength, \
int thread_numbers);
void thread_function_local_histogram_transformation(Mat &outputImage, int rows_thread, \
int cols_thread, int halfSideLength, Mat tempMat, bool isPrint);
void getMeanAndVarianceBaseOnMat(const Mat &inputImage, int array[]);
void LocalWithStatistics(const Mat &inputImage, Mat &outputImage, int sideLength, const double k[]);
#endif