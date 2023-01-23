/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-01-18 21:01:29
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-01-18 21:01:29
 * @Description: notice, all transform about gray value, you can also name it used element value.
 * you must scale the gray value used linear or nonlinear method. because the gray range will not 
 * always form 0 to 255, it might be 0 to m, m might be smaller or bigger than 255. so you should 
 * linear scaling the gray value before displaying the image.
 * so you will find, the function we defined all involved lienar scaling.
 * just like logarithmic and linear scaling, gama and linear scaling. because the linear scaling
 * is necessary for these method.
 * the other scaling method:
 *      if you can show the scaling method used two point. (r1, s1)   (r2, s2)
 *          if r1 = min(element), s1 = 0; r2 = max(element), s2 = (L-1)
 *              this function will be linear scaling.
 *      if you can not show the scaling method used two point. it will be nonlinear scaling function.
 *      if you can use one function show this transform, it will be one general function.
 *      if you can not use one function show this transform,
 *              this function will be piecewise function.
 *              if all piecewise is linear function, it will be linear piecewise function.
 *              if not, it will be nonlinear piecewise function.
 *              of course, the nonlinear piecewise function is more professional.
 * so we should divide this transform into two parts, one part is transform. one part is scalaring.
 * then, we will define all function based on this concept. transform and scaling.
 * transform
 *      linear transform, nonlinear transform(logarithmic and gama transform)
 *      linear transform
 *          identity transform
 *              s = r
 *          reverse transform
 *              s = -r
 *      logarithmic
 *          s = c * log(1 + r)
 *      gama transform
 *          s = c * (r + epsilon)^γ
 * obviously, the gama transform is more generally expression, because the linear transform
 * and logarithmic are all the specific form of the gama transform.
 * scaling
 *      lienar scaling
 *          general scaling:
 *              g = g - min, g = 255 * (g / max)
 *          scaling function based two point. r1, s1, r2, s2
 *      nonlinear scaling.
 * then, all the gray transform is the combination of transform and scaling.
 * you should transform the original gray value first, then scaling the gray value second.
 * then, you will get the efficient what you want.
 * of course, the transform just like logarithmic can also scaling the gray value range of original.
 * but the importance for the transform is it can reduce or increase the details of the original image.
 * so it is the difference between transform and scaling. it means you can use it if you want to
 * show more details of one range gray value in one image if it is less than other gray value range.
***********************************************************************/
#include "../include/grayLevelTransform.h"
/**
 * @Author: weiyutao
 * @Date: 2023-01-16 21:08:57
 * @Parameters: 
 * @Return: 
 * @Description: this function is dedicated to improving
 * the contrasting. used the nonlinear function, it can also be named
 * polynomial function. this function used piecewise function to define.
 * this classic application of piecewise function involved contrast enhancement,
 * threshold processing. we will define this function later.
 */
void contrastEnhance(Mat inputImage, Mat &outputImage) 
{
    return;
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-16 20:39:52
 * @Parameters: 
 * @Return: 
 * @Description: this function can reverse white and blank, just consider it will do it like this,
 * because we it is the most obvious phenomenon. of course, it can also reverse the gray value,
 * but it is not obvious like the white and blank. then, the efficient will be this, 
 * the dark gray region in original will reverse to light gray, the blank region in original
 * will reverse to white, the same concept, the white region in original will
 * reverse to blank.
 * this expression is linear expression.
 */
void reverseTransform(Mat inputImage, Mat &outputImage) 
{
    // reveerse transform expression
    // s = -r + (L - 1)
    double minValue, maxValue;
    Point minLocation, maxLocation;
    minMaxLoc(inputImage, &minValue, &maxValue, &minLocation, &maxLocation);
    outputImage = -1 * inputImage + (maxValue - 1);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-16 20:45:14
 * @Parameters: inputImage: the fourier transform image
 * @Return: 
 * @Description: then, we will consider the nonlinear expression, just like logarithmic
 * the logarithmic can expand the range from little gray value range to bigger
 * gray value range. and narrow the bigger gray value range to
 * little gray value range. it is generally used in fourier transform.
 * how to use it? the fourier transform usually shwo the image used a
 * huge range, like from 0 to 10^6, you will lose many details of the fourier transform
 * image if you want to show the image in your window. so we can use the logarithmic
 * transform to adjust the range. generally, we will adjust the gray range from
 * 0-10^6 to a smaller gray range 0-10. but this gray range is so small
 * that we can not show its more details. so we will use the linear scaling
 * to adjust the gray range from 0-10 to 0-255. why not always use linear scaling?
 * because the logarithmic can keep more details about the original image.
 * so we will use the logarithmetic first, then use the linear scaling.
 * 
 * step1: logarithmic
 *      s = c * log(1 + r)
 *      s is the result gray value, c usually is 1, r is the original element value.
 * step2: linear scaling
 *      g = g - min(g)
 *      g = K[g / max(g)], K = 2^k-1, k is the bit number for image.
 *      generally, k is 8, K is 255.
 * then, you should scaling it.
*/
void logarithmicAndLinearScaling(Mat inputImage, Mat &outputImage) 
{
    // first, you should transform the data type to CV_64UC
    // because you should store the result of logarithmic used CV_64F || CV_32F
    // notice, any error in Mat calculate is usually the reason of data type or data size.
    Mat temp(inputImage.rows, inputImage.cols, CV_64F);
    Mat output;
    inputImage.convertTo(temp, CV_64F);
    double minValue, maxValue;
    Point minLocation, maxLocation;
    minMaxLoc(temp, &minValue, &maxValue, &minLocation, &maxLocation);
    cout << "original min and max: " << minValue << "," << maxValue << endl;
    // this code as follow is equal to inputImage += 1;
    temp += Scalar::all(1);
    // the size and data type of outputImage is equal to inputImage.
    cv::log(temp, output);
    linearScaling(output, outputImage);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-17 21:15:40
 * @Parameters: 
 * @Return: 
 * @Description: we will consider the first linearScalaring method.
 * just like 
 * g = g - min
 * g = K * (g / max)
 * The k is equal to L-1 or 255.
 * we have tested, this scaling will result to the image distortion.
 * then, we will test the linear function used two point defined.
 */
void linearScaling(Mat inputImage, Mat &outputImage) 
{
    Mat temp_linear(inputImage.size(), CV_64F);
    double minValue, maxValue;
    Point minLocation, maxLocation;
    minMaxLoc(inputImage, &minValue, &maxValue, &minLocation, &maxLocation);
    cout << "before scaling: " << minValue << "," << maxValue << endl;
    temp_linear = inputImage - minValue;
    minMaxLoc(temp_linear, &minValue, &maxValue, &minLocation, &maxLocation);
    temp_linear /= maxValue;
    temp_linear *= 255;
    outputImage.create(temp_linear.size(), CV_8UC1);
    temp_linear.convertTo(outputImage, CV_8UC1);
    minMaxLoc(outputImage, &minValue, &maxValue, &minLocation, &maxLocation);
    cout << "after scaling: " << minValue << "," << maxValue << endl;
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-17 21:26:33
 * @Parameters: 
 * @Return: 
 * @Description: this function will defined the linear function used two point.
 */
void linearScalingBaseTwoPoint(Mat inputImage, Mat &outputImage) 
{
    double minValue, maxValue;
    Point minLocation, maxLocation;
    minMaxLoc(inputImage, &minValue, &maxValue, &minLocation, &maxLocation);
    double c = maxValue - minValue;
    double a = 255 / c;
    double b = -255 * minValue / c;
    Mat temp_contrast(inputImage.size(), CV_64F);
    temp_contrast *= a;
    temp_contrast += b;
    outputImage.create(temp_contrast.size(), CV_8UC1);
    temp_contrast.convertTo(outputImage, CV_8UC1);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-18 10:05:39
 * @Parameters: 
 * @Return: 
 * @Description: the more general method is gama or power transform. it can be identity transform
 * , logarithmic transform and power transform based on the different γ.
 * s = c * (r + epsilon)^γ
 * if γ = 1， c = 1. epsilon = 0. --> s = r, it will be identity transform.
 * if c = 1, epsilon = 0, γ > 1, it will be reverse logarithmic transform.
 * if c = 1, epsilon = 0, 0 < γ < 1, it will be logarithmic transform.
 * generally, c = 1, epsilon = 0; 
 * the efficient of this method is, it can make the image darker or lighter. more details, 
 * this method can increse or reduce the details of low gray level or high gray level.
 * if γ > 1, the bigger γ will result to the lighter image. and increase the details of light gray region,
 * reduce the details of dark gray region.
 * if γ = 1, gama transform will be equal to the identity transform.
 * if γ < 1, the smaller γ will result to the darker image. and increase the details of dark gray region, 
 * reduce the details of light gray region.
 * the efficient of param c in gama expression is not obvious.
 * 
 * then, you should notice the difference between transform and scaling.
 * just like the logarithmic transform and linear scaling
 *      the logarithmic transform can also scale the gray value range. but it is efficient
 *      if you want to show more information in a little range gray value. why say it as this?
 *      just like, you have one image that involved 10^6 numbers gray value, but the range is
 *      from 0 to 10, it is the smaller value involved bigger level. you should use logarithmic transform.
 *      becaouse it can reduce the level, it can also means details, but the scaling can just map the range,
 *      just like map the range from 0-10 to 0-255. it will not change the gray level.
 * so you should notice, the level and range. if you want to reduce the details about you image,
 * it can also means you want to reduce the level of your image, you should use transform, but you 
 * should use scaling if you want to map your image gray value range. so we have another standard noun,
 * it is level and range. transform can influence level, and scaling can change range.
 */
void gamaTransformAndLinearScaling(Mat inputImage, Mat &outputImage, double c, double gama) 
{
    Mat temp1, temp2;
    inputImage.convertTo(temp1, CV_64F);
    pow(temp1, gama, temp2);
    temp2 *= c;
    linearScaling(temp2, outputImage);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-19 19:46:02
 * @Parameters: 
 * @Return: 
 * @Description: sometimes, we want to highlight the specific gray value
 * range, how to do it?
 * step1: locate your gray value range.
 * step2: change the value.
 * the step1 is fixed.
 * the step2 mainly contains two method.
 *      set the region you interested used 0 or 255, set the other region used 255 or 0.
 *      set region you interested used bigger or smaller value. the other region remain the same.
 * then, we will finish this method. we named it as gray layered.
 * but we have not found the method of fill the value based on the original 
 * element value of interested region. so we will use the fill the original
 * value for intere1
 * sted region, and fill the other region value used zero.
 * but we can simplify this problem. we can define a spiecewise function.
 * just like the identify function, s = r. you can define the different function
 * for the gray value range what you interested in. 
 * of course, the variable x of this function must be gray value. the interested region
 * can be defined based on coordinates condition or gray value condition. we have defined 
 * the function based on all coordinates of the image. then, we will consider to define the
 * function based on the coordinates condition of the image and the gray value condition of the image.
 * it means you can define the interested region by adding the condition about coordinates.
 * you can also define the interested region by adding the condition about the gray value.
 * they can both define the interested region, only defined it, can you define the different function
 * from the other region of the image. 
 * of course, although defined the interested region used coordinates directly is the most efficient.
 * but based on the gray value is more efficient, because it has not rules, you can define any shape
 * interested region under the condition of without any parameters of the interested region.
 * we will define this function what highlight the interested region besed on the gray level layered
 * as follow.
 * we have defined the function grayLayeredBasedPoints, 
 * it has two function, if mode == 1, it will return one binary image
 * based points you given. if mode == 2, 
 */
void grayLayeredBasedPoints(Mat inputImage, Mat &outputImage, vector<Point> vectorPoints, int mode) 
{
    if (!mode)
    {
        // polylines(inputImage, vectorPoints, true, Scalar(0), 6);
        inputImage.setTo(255);
        // fill the region used bigger value. the other region remain the original.    
        fillPoly(inputImage, vectorPoints, Scalar(0, 0, 0));
        inputImage.copyTo(outputImage);
    }
    else
    {
        // paint polygon lines based some points.
        // polylines(inputImage, vectorPoints, true, Scalar(0), 6);
        // you can also draw line used line function.
        // you can use struct function of Mat to get the region object of one image.
        // just like 
        // Mat roi(image, Range(10, 100), Range(10, 100)), row 10-100, col 10-100
        // Mat roi = image(Range(10, 100), Range(10, 100))
        // you will get the region objetc used this method.
        // you can also use Rect object to store the rectangular region.
                            // of course, you can define vector<vector<Point>> to draw multiple polygon.
        // but how to define the polygon region object? the above method can only define rectangular region.
        // then, how to fill the value based on the original element value?
        // 
        // Mat roi = inputImage(vectorPoints);
        cutImage(inputImage, outputImage, vectorPoints);   
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-21 09:14:05
 * @Parameters: 
 * @Return: 
 * @Description: gray level layered based on the level value. we can traverse each element
 * to judge the condition. this is a simple condition about select the interested gray region.
 * but this method is rough, because we have not used the multithreading, it will be low efficient
 * if you want to handle many image. so we can define the multithreading in your process if you will
 * handle many image. but if the opencv has provided the method about it? we have not found so far.
 * 
 * this function as follow, we have define a piecewise function, min gray value to meanMat is the original
 * gray value, the expression is s = r. meanMat to max gray value is s = 255.
 * it is a simple rule. then, we will learn more complex piecewise function.
 */
void grayLayeredBasedValue(Mat inputImage, Mat &outputImage) 
{
    outputImage = inputImage.clone();
    uchar *matData;
    // you will get three mean value about three channel.
    Scalar mean = cv::mean(outputImage);
    double meanMat = mean[0];
    cout << meanMat << endl;
    for (int i = 0; i < outputImage.rows; i++)
    {
        matData = outputImage.ptr<uchar>(i);
        for (int j = 0; j < outputImage.cols; j++)
        {
            if (matData[j] > meanMat)
            {
                matData[j] = 255;
            }
            else
            {
                matData[j] = 0;
            }
        }
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-23 19:57:26
 * @Parameters: int plane is form 1 to 8
 * @Return: the cooresbonding bit plane image.
 * @Description: this function will define gray layered based on the bit plane. the first function
 * we have defined the gray layered based on the shape involved point. the second function we have
 * defined the gray layered based on the gray value. this function we will defined the gray layered
 * based on the bit plane.
 * a 126 level gray image, the image is consists of b bits. and what is different this and the former
 * function that define gray layered based on the gray value, the former function can highlight the
 * contribution if the gray value, and this function can highlight the contribution of the specific bit.
 * 
 * the 8 bits image involved 8 bit plane, from 1 bit plane to 8 bit plane. the 1 bit plane involved all
 * lowest effictive bits, and the 8 bit plane involved all the highest effictive bits.
 * each bit plane is a binary image. comform the bit plane is 255, the other is 0. the gray value range of n bit plane
 * is from 2^n-1 ~ 2^n - 1, just like 1 bit plane, the gray vlaue is from 0 to 1, the 2 bit plane is from
 * 2 to 3, the 3 bit plane is from 4 to 7, the 4 bit plane is from 8 to 15, the 5 bit plane is from
 * 16 to 31, the 6 bit plane is form 32 to 63, the 7 bit plane is from 64 to 127, the 8 bit plane is from 128 to 255.
 * of course, you can also define the combination of multiple bit plane, just like, you can show 7 bit and 8 bit plane.
 * bit plane transform is a binary transform in fact. of course, this gray level value is also meaningful for
 * image compression.
 * the image compression used bit plane. the concept is we can use the bit plane to reconstruct the original image.
 * it means, you can just store the specific bit plane, it will take up 50 percent of the orginal image memory.
 * we usually use the highest bit plane to reconstruct the original image. you can also use multiple bit plane to
 * do it, the more bit plane you will get the more clear original image.
 */
void grayLayeredBasedBitPlane(Mat inputImage, Mat &outputImage, int plane) 
{
    outputImage = inputImage.clone();
    uchar *matData;
    int start = pow(2, (plane - 1));
    int end = pow(2, plane) - 1;
    if (start == end)
    {
        start = 0;
    }
    for (int i = 0; i < outputImage.rows; i++)
    {
        matData = outputImage.ptr<uchar>(i);
        for (int j = 0; j < outputImage.cols; j++)
        {
            if (matData[j] >= start && matData[j] <= end)
            {
                // notice, Scalar is a specific dedicated to the image data type.
                // uchar is unsinged char. the former is a touple involved three dimension.
                // the last is a unsigned char involved a number, type is unsigned char.
                // you can not cast from Scalar to uchar.
                matData[j] = 255;
            }
            else
            {
                matData[j] = 0;
            }
        }
    }
}


/**
 * @Author: weiyutao
 * @Date: 2023-01-23 21:25:41
 * @Parameters: 
 * @Return: 
 * @Description: this function we will reconstruct the original image used the multiple bit plane.
 * the concept is the bit plane * a const number 2^(k-1), k is the current bit number.
 * the sum all the result. you should map the result to the gray value range from 0 to 255.
 * you should pass the bit number, because we can not get the bit number based on the bit plane Mat.
 * 
 * you can get the most efficient if used more bit plane, but the efficient of lower bit plane
 * is poor, so if you used more smaller bit plane, you will get the poor efficient.
 */
void reconstructImageBasedBitPlane(Mat &outputImage, const map<int, Mat, compareMap> &mapBitPlanes) 
{
    Mat sumMat;
    int i = 0;
    for (map<int, Mat, compareMap>::const_iterator it = mapBitPlanes.begin(); it != mapBitPlanes.end(); it++)
    {
        int n = it->first - 1;
        int k = pow(2, n);
        Mat temp = it->second;
        if (!sumMat.data)
        {
            sumMat.create(temp.size(), CV_64F);
        }
        temp *= k;
        sumMat += temp;
        i++;
    }
    // you can use two method as follow.
    // first you can calculate the mean of sum.
    // second you can map the sum. these two function have the same efficient.
    // sumMat /= i;
    // sumMat.convertTo(outputImage, CV_8UC1);

    linearScaling(sumMat, outputImage);
}