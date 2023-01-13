/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-01-13 12:50:03
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-01-13 12:50:03
 * @Description: this file involed all method to imeplement the image transform,
 * involved used affine matrix and other method.
***********************************************************************/
#ifndef _TRANSFORMUSEDAFFINEMATRIX_H
#define _TRANSFORMUSEDAFFINEMATRIX_H

#include "general.h"
void rotationUsedAffineMatrix(Mat inputImage, Mat &outputImage, int angle, float scale);
void transformUsedOfficialBasedOnThreePoint(Mat src, Mat &outputImage, Size2f firstPoint, Size2f secondPoint, Size2f thirdPoint);
void transformUsedOfficialBasedOnSpecificParam(Mat src, Mat &outputImage, int angle, float scale);
void imageRotationSimple(Mat &inputImage, bool clockwise); 
#endif

