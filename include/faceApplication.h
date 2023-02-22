/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-02-12 17:06:16
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-02-12 17:06:16
 * @Description: this file we will consider some application of face recognize
***********************************************************************/
#ifndef _FACEAPPLICATION_H
#define _FACEAPPLICATION_H
#include "general.h"
#include "../include/linearInterpolation.h"

#if ISOPENDLIB
// some basic utils.
void func(int, void *);
void getTrainDataFromeDir(const string directoryPath, vector<string> &imagePath);
void getFaceSamplesFromMovie(const string moviePath, const string saveDirectory);

// opencv face detect and recognition.
void faceDetectImage(Mat &inputImage, Mat &ouputImage, CascadeClassifier &face_cascade, CascadeClassifier &eye_cascade);
void faceDetectMovie(const string windowName, const string path, CascadeClassifier &face_cascade, CascadeClassifier &eye_cascade);
void faceRecognition(Mat &inputImage, Mat &outputImage);
void faceRecognitionUsedEigenFace(string labelFile, const string predictMoviePath);

// dlib face detect and recognition.
void faceDetectUsedDlib(Mat &inputImage, Mat &outputImage, int mode);
void faceImageRecognitionUsedDlib(const string dirPath, const string targetImage);
void faceMovieRecognitionUsedDlib(const string dirPath, const string targetMoivePath);
#endif

#endif