/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-01-13 12:29:41
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-01-13 12:29:41
 * @Description: involved lienar interpolation, binary linear interpolation, and
 * three linear interpolation.
***********************************************************************/
#ifndef _LINEARINTERPOLATION_H
#define _LINEARINTERPOLATION_H
#include "general.h"
typedef unsigned char uchar;

uchar get_scale_value(Mat &input_image, int x, int y);
Mat scale(Mat &input_image, int height, int width);
uchar get_scale_value_binary(Mat &input_image, float _i, float _j);
Mat binary_linear_scale(Mat &input_image, int height, int width);

Mat resizeImage(Mat &inputImage, float scale);
#endif
