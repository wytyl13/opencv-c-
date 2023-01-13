#ifndef _NOISE_H
#define _NOISE_H

#include "general.h"
void saltPepper(Mat srcImage, Mat &dstImage, int count, int size);
void gaussionNoise(Mat srcImage, Mat &dstImage, float mean, float var);

#endif 
