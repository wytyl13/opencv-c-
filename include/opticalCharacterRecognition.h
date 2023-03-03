/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-03-01 17:28:26
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-03-01 17:28:26
 * @Description: this file we will define some function what can handle
 * some text information in one image. just like you can get the document
 * region from the inputImage. then, you can get each word from the document.
 * actually, we have implemented some function what we will use it in this file.
 * just like the document detected. it can be broken down some steps as follow.
 * step1: detected the four points what can be connected by four lines.
 * step2: affine tranformation rectangle region.
 * step3: binary the rectangle region.
 * step4: detected each word from the result of step3.
 * it is similar to the face detected and face recognition. we can 
 * also name it as doucument detected and word recognition. because 
 * face detected generally is forward position, so we need not affine transform
 * the detected face.
 * 
 * then, we have defined one simple text detected process, but it is not efficient, so we will
 * redefine it. but you should know. just like the face detected, the text detected is also find the
 * four points, draw the rectangle used the four points. the concept of face detected is used the face
 * detected classifier, this classifier will return all the image points in one image. but we have not
 * the ready-made text classifier, so we should train the classifier or define the other method to detect
 * the text in one image. so we will start from the traditional method. we have defined one traditional method.
 * first, detected all the edge in the image. it will return all the closed edge coordinates.
 * second, calculate the area of all the closed edge coordinates. 
 * third, judge the area, if the erea is max and the line numbers of the edge is four.
 * you should return the close edge and segmentate it from the original image.
 * it will be the detected text image what you want to get.
 * but this method is not good. because it is the simple method. then, we will define the other method.
 * the suppose is to get the four points of the text region in original image.
 * then, we will defien one stability method. of course, you can also sanned the word directly.
 * it is the higher application. then, we will define the text detected used the morphology of the image.
 * just like the corrosion expansion. you can get the detected region by enhancing the degree of corrosion 
 * expansion of one image. then, you can get the mincircumscribed rectangle for the corrosion expansion region.
 * notice, the bigger degree of corrosion expansion the more good text detected efficient.
 * so you can find this method is useful for the text detected. but it is not efficient for the blank page.
 * 
 * enother method to detect the text is to use the detected edge line, if they are mutually perpendicular.
 * you can judge it is one page.
 * then, we have understand how to get the four points what used it to rectangle the text region.
 * the corrosion expansion of image is to find the text region directly. it will return the mincircumscibed
 * retangle region of the detected text region. it is dedicated to the text region detected and the text region
 * is absolutely distingush the other region in the image. it means this method will be not efficient if
 * the text region is smaller than other region. similarly, you can not use it to detected the face. because
 * the face region can not be distinguished absolutely from the other region. you can just distingush the face
 * and other region by training the features in face.
 * then, we will start the text recognition from one text region. and we will deep learning the corrosion expansion
 * of image, mincircumscribed rectangle, and other basic knowledge.
 * the text recognition will use the neural network model what is some knowledge about the deep learning.
 * the simple case is SVM model. it is dedicated to train the object classifier, of course, the word what you
 * want to recognize are also the object, face is object, items are also the objects.
 * once we have recognized the words, we will learn how to transform these words to the structured data.
 * it will use some text analysis model what belong to the deep learning. it is irrelevant with opencv.
 * then, let us start to define the funtion that how to transform one text region to the correspond words.
 * 
 * we have finished how to detect the text region from one image. then, we will finish how to recognize
 * each word in from the text region, of course, it is suitable for the original image. because it is 
 * dedicated to recognize the word object.
***********************************************************************/
#ifndef _OPTICALCHARACTERRECOGNITION_
#define _OPTICALCHARACTERRECOGNITION_
#include "general.h"
#include "spatialFilter.h"

class ORC
{
public:
    ORC(){};
    static cv::Mat preProcessing(Mat inputImage) 
    {
        int rows = inputImage.rows;
        int cols = inputImage.cols;
        float scale = 1.0;
        Mat grayImage, resizeImage, blurImage, cannyImage, dilImage;
        cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
        resize(grayImage, resizeImage, Size(cols * scale, rows * scale));
        GaussianBlur(resizeImage, blurImage, Size(3, 3), 3, 0);
        Canny(blurImage, cannyImage, 25, 75);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
        dilate(cannyImage, dilImage, kernel);
        return dilImage;
    }
    static std::vector<Point> getContours(Mat dilImage, Mat &inputImage)
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierachy;
        cv::findContours(dilImage, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // cv::drawContours(inputImage, contours, -1, Scalar(255, 0, 255), 2);
        vector<vector<Point>> conPoly(contours.size());
        vector<Point> biggestArea;
        float maxArea = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            float area = cv::contourArea(contours[i]);
            cout << area << endl;
            if (area > 1000)
            {
                float peri = arcLength(contours[i], true);
                cv::approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
                if (area > maxArea && conPoly[i].size() == 4)
                {
                    maxArea = area;
                    biggestArea = {conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
                }
            }
        }
        vector<vector<Point>> test;
        test.push_back(biggestArea);
        cv::drawContours(inputImage, test, -1, Scalar(0, 0, 255), 2);
        return biggestArea;
    }

    static void drawPoints(Mat &inputImage, vector<Point> points, Scalar color)
    {
        for (int i = 0; i < points.size(); i++)
        {
            circle(inputImage, points[i], 5, color, FILLED);
            putText(inputImage, to_string(i), {points[i].x - 5, points[i].y - 5}, FONT_HERSHEY_PLAIN, 3, color, 3);
        }
    }

    /**
     * @Author: weiyutao
     * @Date: 2023-03-02 12:03:22
     * @Parameters: 
     * @Return: 
     * @Description: reorder the four points based on the shape of z. leftUpper, rightUpper, leftLow, rightLow.
     * notice, you should know the feature of the four point. just like the point as follow.
     * leftUpper (1, 1)
     * rightUpper (11, 2)
     * leftLow (0, 10)
     * rightLow (10, 12)
     * you can find there are one feature for the four points. just like as follow.
     * leftUpper: min(x+y)
     * rightUpper: max(x-y)
     * leftLow: min(x-y)
     * rightLow: max(x+y)
     * so you can reorder the four points based on these feature.
     * but this method has some problem, just like sometimes it will return the same point.
     * just like leftUpper = rightUpper. it will return error result. it means one coordinate
     * satisfied the min(x + y) and max(x + y) at the same time. so it will influence your result.
     */
    static vector<Point> reorderBiggestPoint(vector<Point> points) 
    {
        vector<Point> newPoints;
        vector<int> sumPoints, subPoints;
        for (int i = 0; i < points.size(); i++)
        {
            sumPoints.push_back(points[i].x + points[i].y);
            subPoints.push_back(points[i].x - points[i].y);
        }
        newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
        newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
        newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
        newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
        return newPoints;
    }

    static Mat getWrap(Mat inputImage, vector<Point> points, float w, float h)
    {
        Mat wrapImage;
        Point2f src[4] = {points[0], points[1], points[2], points[3]};
        Point2f dst[4] = {{0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h}};
        Mat matrix = getPerspectiveTransform(src, dst);
        warpPerspective(inputImage, wrapImage, matrix, Point(w, h));
        return wrapImage;
    }

    static vector<Point> getDocumentVectorsFromImage(Mat &inputImage)
    {
        Mat dilImage = preProcessing(inputImage);
        vector<Point> biggestPoint = getContours(dilImage, inputImage);
        return biggestPoint;
    }

    static void documentScanned(Mat &inputImage, Mat &outputImage)
    {
        vector<Point> biggestPoint = getDocumentVectorsFromImage(inputImage);
        vector<Point> reorderBiggest = reorderBiggestPoint(biggestPoint);
        cout << biggestPoint << endl;
        cout << reorderBiggest << endl;
        Mat wrapImage = getWrap(inputImage, reorderBiggest, 420, 596);
        Mat grayWrapImage;
        cvtColor(wrapImage, grayWrapImage, COLOR_BGR2GRAY);
        // edgeStrengthenUsedSobelOperator(grayWrapImage, outputImage, SOBELOPERATORGX, SOBELOPERATORGY, 0);
        threshold(grayWrapImage, outputImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    static void documentScannedBasedonMinAreaRect(Mat &inputImage, Mat &outputImage)
    {        
        Mat resizeImage;
        Mat dilImage = preProcessing(inputImage);
        resize(dilImage, resizeImage, Size(dilImage.cols * 0.5, dilImage.rows * 0.5));
        imshow("123332", resizeImage);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(dilImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point());
        double MaxAreaRRect = 0;
        int SizeContour = 0;
        for (size_t t = 0; t < contours.size(); t++) {
            RotatedRect RRect = minAreaRect(contours[t]);
            double AreaRRect = 0;
            AreaRRect = RRect.size.area();
            MaxAreaRRect = max(MaxAreaRRect, AreaRRect);
        }
        double RRect_degree = 0;
        Mat output = inputImage.clone(); // 这里涉及是否复制数据的问题
        for (size_t t = 0; t < contours.size(); t++) {
            RotatedRect RRect = minAreaRect(contours[t]);
            double AreaRRect = RRect.size.area();
            if (AreaRRect == MaxAreaRRect ) {
                SizeContour = SizeContour + 1;
                // Rotate degree
                RRect_degree = RRect.angle;
                // Draw this rectangle
                Point2f vertex[4];
                RRect.points(vertex);
                for (int i = 0; i < 4; i++) {
                    line(output, Point(vertex[i]), Point(vertex[(i + 1) % 4]), Scalar(0, 255, 0), 2, LINE_8);
                }
            }
        }
        resize(output, output, Size(dilImage.cols * 0.5, dilImage.rows * 0.5));
        imshow("1233333", output);
        Point2f center(inputImage.cols/2,inputImage.rows/2);
        Mat Rotation = getRotationMatrix2D(center,RRect_degree,1.0);
        warpAffine(inputImage, outputImage, Rotation, inputImage.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(255, 255, 255));
        resize(outputImage, outputImage, Size(outputImage.cols * 0.5, outputImage.rows * 0.5));
        imshow("2u9eu2", outputImage);
    }

    ~ORC(){};
};

#endif