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


void func(int, void *);
void faceDetectImage(Mat &inputImage, Mat &ouputImage, CascadeClassifier &face_cascade, CascadeClassifier &eye_cascade);
void faceDetectMovie(const string windowName, const string path, CascadeClassifier &face_cascade, CascadeClassifier &eye_cascade);
void faceRecognition(Mat &inputImage, Mat &outputImage);
void getAllFileFromDirAndCreatTrainData(const string directoryPath, vector<string> &imagePath,\
     const string txtPath, int &count);
void getTrainDataFromeDir(const string directoryPath, vector<string> &imagePath);
void getFaceSamplesFromMovie(const string moviePath, const string saveDirectory);

void faceRecognitionUsedEigenFace(string labelFile, const string predictMoviePath);


void faceDetectUsedDlib(Mat &inputImage, Mat &outputImage, int mode);
#endif