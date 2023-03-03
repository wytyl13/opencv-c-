#include "../include/wordDetect.h"


Mat preprocess(Mat grayImage)
{
    Mat sobel;
    Sobel(grayImage, sobel, CV_8U, 1, 0, 3);
 
    Mat binary;
    threshold(sobel, binary, 0, 255, THRESH_OTSU + THRESH_BINARY);
 
    Mat element1 = getStructuringElement(MORPH_RECT, Size(30, 9));
    Mat element2 = getStructuringElement(MORPH_RECT, Size(24, 4));
 
    Mat dilate1;
    dilate(binary, dilate1, element2);
 
    Mat erode1;
    erode(dilate1, erode1, element1);

    Mat dilate2;
    dilate(erode1, dilate2, element2);
 
    return dilate2;
}


vector<RotatedRect> findTextRegion(Mat inputImage)
{
    vector<RotatedRect> rects;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(inputImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, Point(0, 0));
 
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
 
        if (area < 1000)
            continue;
 
        double epsilon = 0.001 * arcLength(contours[i], true);
        Mat approx;
        approxPolyDP(contours[i], approx, epsilon, true);
 
        RotatedRect rect = minAreaRect(contours[i]);
 
        int m_width = rect.boundingRect().width;
        int m_height = rect.boundingRect().height;
 
        if (m_height > m_width * 1.2)
            continue;
 
        rects.push_back(rect);
 
    }
    return rects;
}

void wordRegionExtract(Mat &inputImage, Mat &outputImage) 
{
    outputImage = inputImage.clone();
    Mat dilation = preprocess(inputImage);
    
    vector<RotatedRect> rects = findTextRegion(dilation);
    
    for(int i = 0; i<rects.size(); i++)
    {
        Point2f P[4];
        rects[i].points(P);
        for (int j = 0; j <= 3; j++)
        {
            line(outputImage, P[j], P[(j + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }
}
