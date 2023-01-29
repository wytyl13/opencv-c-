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
double* getDistribution(Mat inputImage);
void getHistogramMat(double *distribution, Mat &histogramMat);
void histogramTransform(Mat inputImage, Mat &outputImage);
#endif