#ifndef _WORDDETECT_
#define _WORDDETECT_

#include "general.h"
Mat preprocess(Mat grayImage);
vector<RotatedRect> findTextRegion(Mat inputImage);
void wordRegionExtract(Mat &inputImage, Mat &outputImage);

#endif