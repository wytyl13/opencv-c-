#ifndef _FEATUREINIMAGE_
#define _FEATUREINIMAGE_
#include "general.h"
#include "spatialFilter.h"

void hoffmanChainCode();
void boundaryTrackingUsedMoore(Mat &inputImage);
void featureExtract(Mat &inputImage);
void featureDetect(Mat &inputImage);


double calcPCAOrientationUsedOfficial(vector<Point> &pts, Mat &image);
void boundaryTrackUsedOfficial(Mat &inputImage, Mat &outputImage);
void boundaryDetectUsedOfficial(Mat &inputImage, Mat &outputImage);
void featureDetectUsedOfficial(Mat &inputImage, Mat &outputImage);
#endif