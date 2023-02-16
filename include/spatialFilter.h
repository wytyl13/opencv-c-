#ifndef _SPATIALFILTER_H
#define _SPATIALFILTER_H
#include "general.h"

void officialFilterTest(Mat &inputImage, Mat &outputImage, int kernel_number);
void officialImageMixTest(Mat &inputImage1, Mat &inputImage2, Mat &outputImage, float firstWeight);

#endif