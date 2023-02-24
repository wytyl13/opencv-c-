#ifndef _SPATIALFILTER_H
#define _SPATIALFILTER_H
#include "general.h"
#include "grayLevelTransform.h"

#if 1
// official spatial filter test.
void officialFilterTest(Mat &inputImage, Mat &outputImage, Mat kernel);
void officialImageMixTest(Mat &inputImage1, Mat &inputImage2, Mat &outputImage, float firstWeight);
#endif

void operateTwoMatMultiThread(Mat &inputImage, Mat &inputImage_, Mat &outputImage, int symbol);

#if 1
void spatialFilterOperation(Mat &inputImage, Mat &outputImage, Mat kernel);
void spatialConvolution(Mat &inputImage, Mat &outputImage, Mat kernel);
void spatialFilter(Mat &inputImage, Mat &outputImage, Mat kernel, int model);
void spatialFilterUsedSeparatedKernel(Mat &inputImage, Mat &outputImage, Mat kernel, int model);
void medianFilter(Mat &inputImage, Mat &outputImage, int kernelSize);
void getMaskImage(Mat &inputImage, Mat &maskImage, Mat kernel);
void sharpenImageUsedPassivationTemplate(Mat &inputImage, Mat &outputImage, Mat fuzzyOrSmoothKernel, float k);
#endif

#endif