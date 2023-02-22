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
 * but this function can not handle the large error between min and max value in one image.
 * because g=0. it is meaningless.
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


/**
 * @Author: weiyutao
 * @Date: 2023-01-25 20:05:40
 * @Parameters: 
 * @Return: 
 * @Description: get the distribution of the inputImage. pass the inputImage, return the 
 * we can use list in cpp, the index of the list is the gray value. the value in the list is the appearance
 * numbers of the gray value in this image. of course, we can also use the map in cpp container.
 * 
 * so we have learned a method, we do not use the static to modify the local variable, 
 * if you want to return the local variabel in one function, you should malloc it or use the reference parameter
 * to receive the content you want to get.
 */
double* getDistribution(const Mat &inputImage)
{
    // it will return the same address if you used the static to modify the array variable.
    // so we will use malloc function to create this variable. and return it.
    // you should define a pointer first, then malloc a address for this pointer in heap.
    // static double list[256] = {0};
    double *list;
    list = (double *)malloc(sizeof(double) * 256);
    // notice you should distinguish the size and size_t, the former is the numbers in array
    // and the last is the memory size waste for the array.
    // you must initialize the array inside the function, or you will get the error numbers.
    memset(list, 0.0, sizeof(double) * 256);
    const uchar *matRow;
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    double MN = rows * cols;
    for (int i = 0; i < rows; i++)
    {
        matRow = inputImage.ptr<uchar>(i);
        int grayValue = 0;
        for (int j = 0; j < cols; j++)
        {
            grayValue = matRow[j];
            list[grayValue] += 1;    
        }
    }
    // int len = sizeof(list) / sizeof(list[0]);
    // the normalized, in order to handle the problem about the size is different between two image.
    for (int i = 0; i < 256; i++)
    {
        list[i] /= MN;
        // rounded and remain two decimal places.
        // list[i] = (int)((list[i] * 100) + 0.5) / 100.0;
        // list[i] = round(list[i] * 100) / 100;
    }
    return list;
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-25 20:07:31
 * @Parameters: 
 * @Return: 
 * @Description: imshow the histogram. pass the distribution, return a histogram.
 * you will find this histogram is very asymmetric. because we have too much gray value.
 * so it result to the asymmetric happend.
 * 
 * we can name this function as histogram equalization what is a important method
 * for histogram transform. it can enhance contrast.
 */
void getHistogramMatBasedOnDistribution(double *distribution, Mat &histogramMat)
{
    int scale = 2;
    int hist_height = 256;
    histogramMat = Mat::zeros(hist_height, 256 * scale, CV_8UC1);
    Point pt1, pt2;
    double maxValue, currentValue;
    // 64F means double
    // 32F means float
    // 8UC means 8bit unsigned char.
    // Mat_<uchar>对应的是CV_8U
    // Mat_<char>对应的是CV_8S
    // Mat_<int>对应的是CV_32S
    // Mat_<float>对应的是CV_32F
    // Mat_<double>对应的是CV_64F
    // you'd better to use double to store the decimal, or you will get the error number.
    // if you find the error number in your mat object, it must be you have used the float
    // type to store your decimal.
    Mat hist = Mat(256, 1, CV_64F, distribution);
    minMaxLoc(hist, 0, &maxValue, 0, 0);
    for (int i = 0; i < 256; i++)
    {
        currentValue = distribution[i];
        int height = cvRound(currentValue * hist_height / maxValue);
        pt1.x = i * scale;
        pt2.x = (i + 1) * scale - 1;
        pt1.y = hist_height - 1;
        pt2.y = hist_height - height;
        // pt2.y = 500 - i;
        rectangle(histogramMat, pt1, pt2, Scalar::all(255), 1);
    }
}

/**
* @Author: weiyutao
* @Date: 2023-01-30 22:15:12
* @Parameters: inputImage
* @Return: 
* @Description: getHistogramMatBasedOnInputImage
*/
void getHistogramMatBasedOnInputImage(const Mat &inputImage, Mat &histogramMat) 
{
    double *distribution = getDistribution(inputImage);
    getHistogramMatBasedOnDistribution(distribution, histogramMat);
}


/**
 * @Author: weiyutao
 * @Date: 2023-01-31 16:44:45
 * @Parameters: 
 * @Return: 
 * @Description: get the distribution if equalization transform. it is almost equal to the function
 * getCumulativeHistogram we have defined.
 */
double *getEqualizationDistribution(const Mat &inputImage) 
{
    double *distribution = getDistribution(inputImage);
    double *equalizationDistribution;
    equalizationDistribution = (double *)malloc(sizeof(double) * 256);
    // then, transform the distribution.
    double s = 0.0;
    for (int r = 0; r < 256; r++)
    {
        for (int j = 0; j <= r; j++)
        {
            s += distribution[j];
        }
        s *= 255;
        equalizationDistribution[r] = round(s);
        s = 0.0;
    }
    return equalizationDistribution;
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-25 18:20:28
 * @Parameters: 
 * @Return: the image after doing histogram equalize transformation.
 * @Description: histogram equalize transformation processing. 
 * p(rk) = h(rk) / MN = nk / MN
 * M N is the numbers of rows and columns, for all p value of k, sum p(rk) = 1;
 * the component of p(rk) is the probability estimates for the gray value in the image.
 * the histogram is the basic operation in the image processing. because it is simple and 
 * suitable for hardware implementation quickly. so the histogram technology is a popular tool
 * for the real-time image processing. the shape of the histogram is related with the appearance
 * of the image. we can image the relationship between histogram and the appearance of the image.
 * the horizontal axis of the histogram is the gray value. the histogram will foucuse on the left
 * if the picture is the dark image. the histogram will foucuse on the right if the picture is the
 * bright image. the histogram will foucuse on the middle if the picture is low contrast.
 * the histogram will be uniform distribution if the picture is high contrast.
 * so h(rk) is the numbers of the gray value appear in the image.
 * h(rk) / MN is the probabilty of the gray value appear in the image.
 * 
 * the concept of histogram.
 * just like a 3 bit image, L = 2^3 = 8. L - 1 = 7. a 64*64 = 4096 image.
 * we can get the gray scale distribution for this image.
 * gray value       numbers         probability
 * 0                790             790/4096=0.19
 * 1                1023            0.25
 * 2                850             0.21
 * 3                656             0.16
 * 4                329             0.08
 * 5                245             0.06
 * 6                122             0.03
 * 7                81              0.02
 * the normalized image histogram. it is a expression. histogram equalization.
 * s_k = T(r_k) = (L - 1) * Σ(j = 0~k) p_r(r_j)
 * s0 = 7*p_r(r0) = 7 * 0.19 = 1.33 -rounded-> 1 
 * s1 = 7*[p_r(r0)+p_r(r1)] = 7*(0.19+0.25) = 3.08 -rounded->3
 * s2 = 5
 * s3 = 6
 * s4 = 6
 * s5 = 7
 * s6 = 7
 * s7 = 7
 * we can see, the mapping for this expression is one to many mapping.
 * the gray value is monotonous increasing.
 * the mapping of the gray value has two cases.
 * first is monotone increasing function, it means one to many mapping.
 * second is strictly monotone increasing function, it means one to one mapping.
 * it is monotone increasing function in this case.
 * then we can get another hostogram based on the mapping gray value.
 * 1        790                 790         0.19   
 * 3        1023                1023        0.25
 * 5        850                 850         0.21
 * 6        656+329             985         0.24
 * 7        245+122+81          448         0.11
 *
 * the histogram is the approximate of the probabilty density function.
 * this case used normolized image histogram. it will cover the border gray range after
 * the equalization of the image. it means we can enhance the image contrast by using
 * normolized image histogram. this is a specific histogram transform.
 * you can see the probabilty will be balanced for each gray value, and the gray value 
 * will be similar to the original gray value.
 * and a special feature is that, no matter what type image you given, you can get the same high contrast
 * image. it means, if you give some images that the different darker or brightness of one image.
 * carrying out the transformation of histogram equalization to them. you can get the different
 * histogram for the image after transformation, and the result image is euqal.
 * the didfferent darker or brightness image of one same image, you can get the same high contrast
 * image by doing the histogram equalization transformation for them. and the histogram for all high
 * contrast you have got are different. it can also mean that the histogram of the same image may be different.
 * it means the different histogram of one image can show the same contrast picture.
 * why? because all the image just has the different contrast for one same image, so you can get the same
 * high contrast image by using histogram equalize transformation. if this condition do not reach, you can not
 * get this effection.
 * 
 * of course, you can define histogram transform used the official function that opencv has defined.
 * it is equalizeHhist in opencv. 
 * but it may not be appropriate used histogram equalization in some appication. because the histogram equalization
 * will generate the not sure histogram. but sometimes we may want to generate the specific shape histogram.
 * it can be name as histogram matching. the difference between the histogram equalization and histogram
 * matching is the former will generate the unknown histogram based on the known histogram transformation function.
 * the last will generate the known shape histogram based on the changeable histogram transformation function.
 * the former the transformation funcition is the only, the last transformation function need to be calculated
 * according to the known shape of the histogram. the last method can be named the histogram matching.
 * 
 * the histogramEqualizeTransformation function is invalid for the brightness image. because it can just
 * improve the details about the low gray level value. so we should define the histogramMatchingTransformation
 * function.
 */ 
void histogramEqualizeTransformation(const Mat &inputImage, Mat &outputImage) 
{
    // you should get the histogram of the original image.
    double *distribution = getDistribution(inputImage);
    double transformValue[256] = {0};
    // then, transform the distribution.
    double s = 0.0;
    for (int r = 0; r < 256; r++)
    {
        for (int j = 0; j <= r; j++)
        {
            s += distribution[j];
        }
        s *= 255;
        transformValue[r] = round(s);
        s = 0.0;
    }
    // change the original gray value.
    outputImage = inputImage.clone();
    int rows = outputImage.rows;
    int cols = outputImage.cols;
    uchar *rowMat;
    for (int i = 0; i < rows; i++)
    {
        rowMat = outputImage.ptr<uchar>(i);
        for (int j = 0; j < cols; j++)
        {
            rowMat[j] = (uchar)transformValue[rowMat[j]];
        }
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-30 11:57:25
 * @Parameters: 
 * @Return: 
 * @Description: define the get cumulative histogram function. we will use the former 
 */
double *getCumulativeHistogram(const Mat &inputImage) 
{
    // get the distribution of the original image first.
    // notice the difference between pointer variable and array variable.
    // distribution is a pointer variable here, so you can not calculate the size of the array used sizeof
    double *distribution = getDistribution(inputImage);
    double s = 0.0;
    double *transformValue;
    transformValue = (double *)malloc(sizeof(double) * 256);
    // int len = sizeof(distribution) / sizeof(*distribution); this will return 1, because the size of pointer is 8 bytes.
    // and the double is also 8 bytes. 8 / 8 = 1;
    // we will write a fixed length numbers. if you want to define a dynamic method, you should modify the getDistribution function.
    // second, you should calculate the cumulative histogram.
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            s += distribution[j];
        }
        transformValue[i] = s;
        s = 0.0;
    }
    return transformValue;
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-30 11:46:11
 * @Parameters: use the const to modified the former two parameters, they can not be changed.
 * and use reference to modified all parameters, it means we can save the memory that parameter 
 * wasted the run the stack. notice, the inputImage is the histogram equalization image here. objectImage
 * is any gray image.
 * @Return: outgoing param outputImage.
 * @Description: we should understand the different between histogram equalization and histogram
 * matching, the former will improve the contrast, but it did not consider the image
 * color evenly, it means, the histogram equalization will only make the contrast of the image
 * clearly, but it did not consider the truth of the image. the gray value will drop a bluff.
 * but the last method histogram matching will consider the imange color enenly on the basis
 * of the histogram equalization.

 * then, how to define the histogram matching on the basis of the histogram equalization?
 * you can use two method:
 *     first, you can get the histogram based on one image. it means you can histogram equalization
 *     from the histogram of one image to the histogram of another image.
 *     
 *     second, you can give a fixed histogram function.
 * then, the first method is generally used. because it is simple and feasible.
 * the generally scenario for histogram matching is, if you want to improve the contrast of one image, 
 * you can do histogram equalization for the original image, if you want to maintain uniform color image,
 * you should do the histogram matching on the basis of the result after doing histogram equalization.
 * then, we will construct this application, remain the image color evenly and to increase the contrast of the image.
 * 
 * in the ideal state, histogram equalization realize the image gray uniform distribution. it is useful to increase the
 * the contrast of the image, generally, the increasing of image contrast means the increasing about the brightness.
 * but in practical application, we do not want the image histogram has the uniform distribution of the whole sometimes.
 * it means we want to a gentle changeable for image gray. then we will give a shape of histogram, and design a histogram
 * transformation function based on the original histogram and the histogram we give. 
 * 
 * then, the original histogram is usually the histogram after equalization, the histogram we give is 
 * usually got from a image. so it means the param of function we will define involved the original image and
 * the image we want to get histogram from it. and the outputImage as the reference pass parameter to store the result
 * after the histogram transformation.
 * so this function we can describe as follow.
 *     what we will do is with reference to the gray level distribution of the objectImage to transform the inputImage to 
 *     the outputImage. we will get the specific gray level distribution not the uniform distribution. the sepecific
 *     gray level distribution we will get it from the objectImage.
 * then, the main content of the function is how to transform from a fixed histogram to a fixed histogram.
 * then, we will describe this transformation process.
 * 1 histogram equalize the histogram of the inputImage. s = T(r), r is the original gray, s is the gray after histogram equalizing for the original histogram.
 * 2 histogram equalize the regulation histogram. v = G(z), z is the gray value of the regulation histogram. v is the gray after histogram equalizing for the regulation histogram.
 * 3 because the regulation and original histogram are all the histogram of the same original image, so the s is euqal to v.
 *     it means z = G^-1(v) = G^-1(T(r)), the r is known, so we can get z used this expression.
 * the more details is as follow.
 * s_k=T(r_k)=(L-1) * ∑i=0_k P_r(r_k)
 * v_k=G(z_m)=(L-1) * ∑j=0_m P_z(z_m)
 * then you can get the function map from s to z. you should find the minimize(∑i=0_k P_r(r_k) - ∑j=0_m P_z(z_m))
 * minimize(∑i=0_k P_r(r_k) - ∑j=0_m P_z(z_m))
 * the k is the original gray value, the m is the regulation histogram gray value.
 * known the k, the m of minimize this function is the mapping. it menas k-->m, m meet minimize(∑i=0_k P_r(r_k) - ∑j=0_m P_z(z_m))
 * in order to understand, we will define some new word.
 * ∑i=0_k P_r(r_k) is the original cumulative histogram.
 * ∑j=0_m P_z(z_m) is the regulation cumulative histogram.
 * minimize(∑i=0_k P_r(r_k) - ∑j=0_m P_z(z_m)), m is the gray value that minimize the difference between
 * the two cumulative histogram.
 * so we will define this histogram matching transformation based on the concept above.
 * you can find from this function, if you give a low gray value image, you will get a low gray value mapping.
 * or you will get a equal or high gray value mapping.
 * but the efficient of this function is bad for the gray image. because its color is single.
 * unless you use the pure white or pure blank image you can see the different gray mapping.
 * or you will get the equal mapping Mat to the original gray value.
 
 * then, we have found a problem. no matter select which one picture as object image. always get a replica
 * of the original image. it means the function histogramMatchingTransformation we defined is not working.
 * why? we found the problem is the function getDistribution in getCumulativeHistogram function is error.
 * no matter you pass any inputImage, you will get the same distribution. so we will modify this error.
 
 * then, we have found the problem in getDistribution function, because we define the list in the function, and 
 * use the static to modify it, so this function will return the same pointer created in this function stack.
 * so we pass the different image, this function will always return the same distribution. how to fix it?
 * you can use the reference pass parameters to instead the return type. or you can malloc this pointer in
 * the function. because you have modified this variale list used static, so thsi variable will be the fixed
 * global variabel. and the same address in this function will show one fixed information in the same time.
 * we will use the malloc method to handle this problem. because malloc function will create the address in the heap.
 * the static modified the variable in function will create the address in stack, and the stack will be fixed in
 * the same process, and this address will not free after the function run end if you have modified it used static.
 * so it means we will create this array used malloc function. the function of using malloc is create the variable 
 * in heap when the function run in the stack.
 * 
 * you should notice, the histogramMatchingTransformation function is based on the histogram equalization function.
 */

void histogramMatchingTransformation(const Mat &inputImage, const Mat &objectImage, Mat &outputImage) 
{
    // you should get the cumulative histogram for the original image and object image.
    // we will use the get cumulative histogram function defined by ourselves.
    double *cumulativeDistributionInputImage = getCumulativeHistogram(inputImage);
    double *cumulativeDistributionObjectImage = getCumulativeHistogram(objectImage);
    // calculate the difference between two cumulative histogram. we will use a two dimension array to store it.
    // each cumulative has 256 element, the difference for each element is 256*256.
    double difference[256][256];
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            difference[i][j] = fabs(cumulativeDistributionInputImage[i] - cumulativeDistributionObjectImage[j]);
        }
    }

    // construct the gray level value mapping table.
    Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
    {
        double min = difference[i][0];
        int index = 0;
        for (int j = 1; j < 256; j++)
        {
            if (min > difference[i][j])
            {
                min = difference[i][j];
                index = j;
            }
        }
        // lut.at<uchar>(i) = static_cast<uchar>(index);
        lut.at<uchar>(i) = static_cast<uchar>(index);
    }
    // cout << lut << endl;
    // then, you should mapping the gray table you have got to change the original gray value.
    // of course, you can also use the function define by yourself. 
    // we used the LUT function that official defined.
    LUT(inputImage, lut, outputImage);

    // of course, you can use the LUT function to map the original gray value.
}


/**
 * @Author: weiyutao
 * @Date: 2023-01-31 15:04:51
 * @Parameters: 
 * @Return: 
 * @Description: util now, the histogram processing we have contacted are all global. the global processinig
 * method is suitable for the overall enhancement of image. it will be failure to handle the increase in the local area.
 * the solution is designed the transformation function based on the neighborhood of the gray level distribution.
 * the differenece between global and local histogram is the former is based on the whole gray value of the image.
 * the last is based on the local region gray value of the image. the former can not handle the situation that
 * gray value of local region is highlight than the other region. just like the people in the night, the gray 
 * value of people region is bigger than the other region, or the local region is very small relative to the whole image.
 * 
 * of course, the local histogram transformation must specified the local size.
 * generally, the gray value range from 0 to 100 is dark. range from 100 to 150 is low contrast image.
 * range from 150 to 250 is light image. the histogram equalization is mapping the gray value from 
 * range (100,150) to range(0,255)
 * 
 * the application of local histogram transformation is we can enhance the contrast of local region in the image.
 * it will enhance the highlight of the local region and will not reduce the highlight of the other light region.
 * you will not acheive this effect if you used global histogram transformation.
 * just like the image as follow
 * 0 0 0 0 255 255 255 255 255 255
 * 2 188 188 188 255 255 255 255 255 255
 * 188 188 188 255 255 255 255 255 255
 * 0 0 0 0 255 255 255 255 255 255
 * 0 0 0 0 255 255 255 255 255 255
 * 0 0 0 0 255 255 255 255 255 255
 * if you want to enhance the left upper region contrast, you will reduce the other high region contrast
 * if you used the global histogram transformation. but it will not happen if you used the local
 * histogram transformation.
 */
void histogramTransformationLocal(const Mat &inputImage, Mat &outputImage, int sideLength)
{
    int halfSideLength = sideLength / 2;
    int rows = inputImage.rows;// hight
    int cols = inputImage.cols;// width
    outputImage = inputImage.clone();
    double *tempMatEqualizationDistribution;
    for (int row = halfSideLength; row < (rows - halfSideLength); row++)
    {
        for (int col = halfSideLength; col < (cols - halfSideLength); col++)
        {
            // scanf the region element.
            Rect rect = Rect(col - halfSideLength, row - halfSideLength, sideLength, sideLength);
            Mat tempMat = outputImage(rect);
            // then calculate the equailization histogram of the local region rect, it is the tempMat.
            // we will calculate the mapping table about the gray value in global histogram transformation.
            // but now we are handling the local histogram. so we just need to mapping the center of the 
            // local region to the new gray value we have calculated used histogram equalization transformation method.
            tempMatEqualizationDistribution = getEqualizationDistribution(tempMat);
            int index = (int)(tempMat.at<uchar>(tempMat.rows / 2, tempMat.cols / 2));
            outputImage.at<uchar>(row, col) = tempMatEqualizationDistribution[index];
        }
    }
}

void thread_function_local_histogram_transformation(Mat &outputImage, int rows_thread, \
int cols_thread, int halfSideLength, Mat tempMat, bool isPrint = false) 
{
    for (int row = halfSideLength; row < (rows_thread - halfSideLength); row++)
    {
        for (int col = halfSideLength; col < (cols_thread - halfSideLength); col++)
        {
            // the first and second param of Rect is the original point coordinated, the third and fourth
            // of Rect is the width and height of the Rect.
            Rect rect = Rect(col - halfSideLength, row - halfSideLength, halfSideLength * 2, halfSideLength * 2);
            Mat kernelMat = tempMat(rect);
            double *tempMatEqualizationDistribution = getEqualizationDistribution(kernelMat);
            int index = (int)(kernelMat.at<uchar>(kernelMat.rows / 2, kernelMat.cols / 2));
            tempMat.at<uchar>(row, col) = tempMatEqualizationDistribution[index];
            if (isPrint)
            {
                cout << row << " " << col << endl;
            }
        }
    }
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-01 09:46:07
 * @Parameters: 
 * @Return: 
 * @Description: then, we will perfect this local histogram transformation by
 * adding the method of thread. if you want to show the perfect result, you should pass the correct param.
 * the shape of the image should suitable for the sideLength and thread_number, it means the shape you defined
 * should be as follow. 360*700, sideLength = 4, thread_numbers = 4.
 * 360 / 2 = 180; 180 / 4 = 45. 700 / 2 = 360, 360 / 4 = 90. you should ensure the it can be divisible.
 * not the remainder. then, we will reshape the image to 360*700 to test the efficience.
 * but this local histogram function has a letal problem. no matter how to optimize this process,
 * the result image always has some highlights in the middle of the image. because you process is not
 * continuous if you added the thread, because the element mapping is not continuous. but this problem will
 * not exists if you do not use the thread. just like as follow.
 * 1 0 0 1 1 1 0 0
 * 1 1 1 0 0 1 1 1
 * 1 1 1 0 1 0 1 1
 * 0 1 0 1 0 1 1 0
 * 
 * if you want to handle this process used general method. just like 2*2 kernel. the mapping is sum.
 * you will get a 2*6 dimension matrix from a 4*8 matrix, notice, the former change can influence the last change.
 *    7 11 15 20 26 31
 *   13 24 37 56 81 113
 * 
 * if you use the thread to kernel this process. you will get a 3*6 dimension matrix, the result is as follow.
 * if you want to scan all element, you use the method as the follow function. because the you used the thread, so
 * the former change will not influence the last value in some coordinates.
 *   7 11 15 20 6 11
 *   7 12 15 19 6 13
 * you can find, if you used thread, some coordinates just like the different value coordinates will
 * be the highlights.
 * how to fix the problem? you can just set to divide the row not the column, to handle the highlights in
 * column. temporarily no the method to handle all the problem.
 * this method is to divide this original image into four uniform. then scan the image from left upper corner
 * to the low right corner. it will happend to this highlight problem.
 * you can try to scan from the four corner and scan end to the center of the image. this problem will be dropping.
 * but this problem will be still exists.
 * 
 * then, you should distinguish the difference between width, height and coordinates.
 * just like a 2*2 kernal in the leftUpper of the image. it involved
 * 0,0 1,0 2,0
 * 0,1 1,1 2,1
 * 0,3 1,3 2,3
 * it involved 9 coordinates, the width is 2, the height is 2, the area is 2*2=4
 * but the coordinates it involved is not 4, but the 9.
 * 
 * 
 * you can try scan each element based on the row or columns, you can also scan each element based on the diagonal.
 * then, this problem we will optimize it at last.
 */
void histogramTransformationLocalThread(const Mat &inputImage, Mat &outputImage, int sideLength, \
int thread_numbers) 
{
    if (thread_numbers % 2 != 0 && thread_numbers > 12)
    {
        sys_error("you should input an even number and less than 12");
    }

    int w = (thread_numbers / 2) > 4 ? 4 : thread_numbers / 2;
    int h = thread_numbers / w;
    int halfSideLength = sideLength / 2;
    int rows = inputImage.rows;// hight
    int cols = inputImage.cols;// width
    outputImage = inputImage.clone();
    // if you want to add the pthread in this function, you should make an issue of the for loop
    // first, you should create the recall function dedicated to using in this function.
    // then, you should calculate the thread numbers you should create.
    // then start each thread.
    // at the end, free the thread.
    // the idea to handle this problem is we will divide the input image into four quartering.
    // it means we will create four thread to handle this function.
    int rows_thread = rows / h;
    int cols_thread = cols / w;
    int width, height;
    // x, y means the coordinates.
    // you should find the leftupper pointer you will handle the small mat in each thread.
    // because we used the for loop create the thread, so we should use a vector to store it.
    // you should use emplace_back not the push_back.
    vector<thread *> vectorThreads;
    for (int i = 0, x = 0, y = 0; i < thread_numbers; i++, x += cols_thread)
    {
        if (i % w == 0 && x != 0)
        {
            x = 0, y += rows_thread;
        }
        // find the region that relative to the right and bottom.
        // define the different shape region. over more area. or you will get the 
        // obvious differences in the result image.
        // find the lowerRight region.
        if (i % w == 1 && (i == (thread_numbers - 1)))
        {
            width = cols_thread;
            height = rows_thread;
        }
        else if(i >= (thread_numbers - w))
        {
            // height is original, width should add sideLength - 1;
            width = cols_thread + sideLength - 1;
            height = rows_thread;
        }
        else if((i % w) == 1)
        {
            // width is original, height should add sideLength - 1;
            width = cols_thread;
            height = rows_thread + sideLength - 1;//
        }
        else
        {
            width = cols_thread + sideLength - 1;
            height = rows_thread + sideLength - 1;            
        }
        cout << width << " " << height << endl;
        Rect rect = Rect(x, y, width, height);
        Mat tempMat = outputImage(rect);
        thread *thread_pointer = new thread (thread_function_local_histogram_transformation, ref(outputImage), rows_thread, \
        cols_thread, halfSideLength, tempMat, false);
        vectorThreads.push_back(thread_pointer);
    }
    for (long long unsigned int i = 0; i < vectorThreads.size(); i++)
    {
        thread *thread_i = vectorThreads[i];
        if (thread_i != NULL)
        {
            thread_i->join();
            delete thread_i;
            thread_i = NULL;
        }
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-03 10:51:19
 * @Parameters: 
 * @Return: 
 * @Description: if you want to return a array in one funciton, you'd better use the pointer parameter.
 * this pointer defined in the main function. if the process run end, the pointer will be freed automatic.
 * you would hand relese this pointer if you want to return the array, because you must malloc inside the function.
 * it will create the pointer in heap. and you can also use static to modify the pointer inside the function.
 * but you will have a problem that you will get the same content if you call this function in the same function.
 * so you'd better use the pointer parameter.
 * m = Σi=0_(L-1) r_i * p(r_i)
 * variance = Σi=0_(L-1) (r_i - m)^2 * p(r_i)
 */
void getMeanAndVarianceBaseOnMat(const Mat &inputImage, int array[]) 
{
    double *distribution = getDistribution(inputImage);
    double tempValue = 0.0;
    int mean;
    for (int i = 0; i < 256; i++)
    {
        tempValue += (i * distribution[i]);
    }
    mean = (int)tempValue;
    array[0] = mean;
    tempValue = 0.0;
    for (int i = 0; i < 256; i++)
    {
        tempValue += (pow((i - mean), 2) * distribution[i]);
    }
    array[1] = (int)tempValue;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-02 10:55:23
 * @Parameters: 
 * @Return: 
 * @Description:  then, we will understand the concept of moment.
 * N moments is defined as the integral of the product of the power and probability density function
 * about one variable. just like μ_n = Σi=0_m r_i^n * p(r_i), p(r_i) is the probability desity function
 * about the r_i. r_i is the variable, the range of r_i is (0, m), it means the gray value in the digital
 * image processing, the m is L-1 in for one image.
 * based on the above content. we can define N moments of the gray value r relative to the mean is as follow.
 * μ_n = Σi=0_(L-1) (r_i - m)^n * p(r_i), m = Σi=0_(L-1) r_i * p(r_i).
 * it just is equal to the expressim m = (Σi=0_(L-1) r_i) / (L-1), it is the mean of r_i in fact.
 * the variance of r_i is μ_2 = Σi=0_(L-1) (r_i - m)^2 * p(r_i), it is the 2 moments of the (r_i - m).
 * this expression about variance or standard deviation is the measure of the image contrast.
 * generally, the mean and variance is the tools to enhance the image that we should consider.
 * it involved global mean, variance and local mean, variance. but the more powerful application is 
 * in local handle. the local mean and variance is calculated based on the neighborhood of one element
 * in one image.
 * then, we can define the expression about the mean of the local region.
 * m_Sxy = Σi=0_(L-1) r_i * p_Sxy(r_i)
 * m_Sxy is the mean of the local region in one image.
 * L is the max possible numbers of the gray value in the local region of the image. it usually is 256 in global region.
 * but in the local region, but the possible numbers of the gray value is determined by the shape of the region
 * in local region. it is not set up for the global image. the possible numbers of the gray value in global
 * image is 256, but it is determined the area about the region in local region. L is equal to 9 if the local region is 3*3
 * because the possible numnber of the gray value in the region is 9. so the L is 9.
 * p_Sxy(r_i) is the probability density function about the each element for the local region.
 * this 3*3 here means the numbers of element. the row has 3 element, the column has 3 element.
 * it involved 9 element. so the probability numbers of gray value is 9. notice, the difference 
 * between element numbers and area.
 * then, we will unified the dimension. the m*n means the the row numbers are m, the column numbers are n.
 * just like the kernel dimension. it generally is an even number. the even number in computer science
 * means the odd number. only the odd numbers has the center. just like 3*3
 * 1 2 3
 * 4 5 6
 * 7 8 9
 * the center is 5, the coordinates is (1, 1), (2/2, 2/2)
 * the dimension in kernel means the length. just like 2*2, it means the row length is 2, the cloumn length
 * is 2, the numbers of gray value in row is 3, the numbers of gray value in column is 3.
 * and the index started from 0, so we can use the length dimension to show the kernel size.
 * notice the difference of dimension information betweem kernel and image.
 * 
 * 
 * let's return the mean and variance of the neighborhood element.
 * the variance of the neighborhood element expression is 
 * variance(Sxy) = Σi0_(L-1)(r_i - m_Sxy)^2 * p_Sxy(r_i)
 * L-1 is 256, it means the probability numbers of gray value in the image. it is a fixed container.
 * the numbers of the container are always 256, the difference between difference dimension region is 
 * the difference region has the effective container different numbers. the effective container is nonzero.
 * the invalid container is zero. just like the 2*2 dimension region. it involved 3*3 numbers element.
 * so the probability numbers of the gray value in this region are 9. so it will has 9 containers
 * is effective, each effective container involved the numbers of the gray value occured in the local region.
 * because the local region may be up to occure nine different gray value. so the container numbers 9
 * is the max numbers. if the local region has 9 different gray value, the value will be 1 in each container.
 * or the effective container numbers will be reduced.
 * 
 * the expression local region means is the messure of the average gray value for the neighborhood of the center gray value.
 * the expression local region variance is the messure of the contrast for the neighborhood of the center gray value.
 * the application for these two indicators that are closely related to the image appearance is we can develope simple
 * and strong rules of image enhancement.
 * 
 * how to ocnstruct this relationship based on the contrast enhancement and these two indicators.
 * we will distinguish the darker and darker relative to the global image for each gray value.
 * compare the mean gray value of local region and the global region. the local region is Sxy.
 * m_Sxy represent the mean gray value of local region and m_G represent the mean gray value of global 
 * region. we can add two parameters. just like as follow.
 * k0*m_G <= m_Sxy <= k1*m_G, the range value of k0 and k1 is from 0 to 1, and k0 is less than k1.
 * so if the m_Sxy is in the middle of k0*m_G and k1*m_G, this coordinates is adjustable. it means
 * we can find the adjustbale coordinates that can enhance the contrast of image by this simple expression.
 * of course, we can also define the paramters based on the idea that ourselves.
 * just like we can define the k0 is 0, and k1 is 0.25, it means the important region we interested in is
 * the coordinates that gray value the range is from 0 to 1/4 of the global mean gray value.
 * this is how to find the coordinates that we interested in. but the interested region must can be adjustble?
 * we should judge it based on the compared result that the variance of the global region variance and
 * local region variance. it is same to as the mean, you should define two parameters to judge.
 * just like k2*variance_G <= variance_Sxy <= k3*variance_G. if the condition is suitable for the expression.
 * we can adjust the coordinates. this expression can judge the coordinates.
 * 
 * with those tools above, we can do the local histogram transformation more flexible. just like you can 
 * do it by adding some condition, not for all. 
 * for example:
 * just like the local region is blank background. and the other region is highlight background.
 * the local region means to enhance the regional. the enhance means to increasing the contrast.
 * the range of gray value in global region is from 0 to 228.
 * the range of gray value in local region is from 0 to 10.
 * you can find that the gray value difference between local and region is very big.
 * the m_G is equal to 161, variance_G is 103. of course, you can image that m_Sxy and variance_Sxy
 * is far less than m_G and variance_G. so the parameter you can define smaller, just like from 0 to 0.1.
 * this parameters can judge the condition that the rate about m_Sxy, variance_Sxy and m_G, variance_G.
 * if the true value for m_Sxy and variance_Sxy within the range of proportion. you can do mapping the 
 * current scaned gray value used the center gray value of the keneral Mat in the mapping distribution.
 * of course, the mapping distribution can calculated used equalization and matching method.
 * the m_G, m_Sxy can messure the average gray value. variance_G and variance_Sxy can messure the 
 * contrast of the image. you can decide whether to do change by comparing these two indicators between 
 * local region and global region. but the local region means what? means the kernel scaned region or 
 * the interested region? of course, it is the kernel scaned region. becasue we can not define the interested
 * region, so we define some parameters based on the average and variance of the gray value. in order
 * to judge the interested region by some condition. so you should known the relative about the kernel region and local region.
 * the kernel region is scaned region. the local region is the interested region. you can define the local
 * region by adding some judge condition into the kernel region.
 * 
 * you can image that the local histogram trasnformation function, no matter the equalization or matching transformation.
 * the purpose is we want to scan kernel size region, and calculate the mapping based on the kernel to change 
 * the original gray value. but the global transformation is to map the gray value by calculated the mapping
 * based on the global region. so the difference between local transformation and global transformation is 
 * the former can mapping the gray value more continue. because the former changed gray value by kernel can 
 * influence the last. but the global transformation can not do it.
 * 
 * but no matter the local or global transformation, they can all transforme all gray value. we can not 
 * determine which region we want to remain the original value. so we introduced the m_G, variance_G, m_Sxy
 * and variance_Sxy, aim to judge which region we want to change, which region we do not want to change.
 * that is all.
 * then, we will define the function about hitogram transformation used histogram statistics.
 * actually very simple, it is to add some condition when you scaned the kernel.
 * 
 * finally, how to change the gray value if you have found the interested region?
 * the equalization transformation or matching transformation are all good method.
 * but these two method are all not suitable here.
 * the simple and effective method is multiplied by a fixed constant.
 * the constant generally is equal to the max gray value about global region divided the max gray
 * value about local region.
 * 
 * notice, the scaned kernel size, it means sideLength in this function, is relatived to the
 * k0, k1, k2, k3. because the kernel size is smaller, the corresponding mean and variance will
 * be changed.
 */
void LocalWithStatistics(const Mat &inputImage, Mat &outputImage, int sideLength, const double k[])
{
    int array_glocal[2] = {0};
    getMeanAndVarianceBaseOnMat(inputImage, array_glocal);
    int m_G = array_glocal[0];
    int variance_G = array_glocal[1];
    int m_Sxy, variance_Sxy;
    double k0 = k[0];
    double k1 = k[1];
    double k2 = k[2];
    double k3 = k[3];
    int halfSideLength = sideLength / 2;
    int rows = inputImage.rows;// hight
    int cols = inputImage.cols;// width
    outputImage = inputImage.clone();
    int array_kernel[2] = {0};
    double maxValueGlobal, maxValueKernel;
    minMaxLoc(outputImage, 0, &maxValueGlobal, 0, 0);
    // calculate the m_G and variance_Sxy.
    // we will define the function about calculated the m and variance out of the function.
    for (int row = halfSideLength; row < (rows - halfSideLength); row++)
    {
        for (int col = halfSideLength; col < (cols - halfSideLength); col++)
        {
            // scanf the region element.
            Rect rect = Rect(col - halfSideLength, row - halfSideLength, sideLength, sideLength);
            Mat tempMat = outputImage(rect);
            minMaxLoc(tempMat, 0, &maxValueKernel, 0, 0);
            // then calculate the equailization histogram of the local region rect, it is the tempMat.
            // we will calculate the mapping table about the gray value in global histogram transformation.
            // but now we are handling the local histogram. so we just need to mapping the center of the 
            // local region to the new gray value we have calculated used histogram equalization transformation method.
            getMeanAndVarianceBaseOnMat(tempMat, array_kernel);
            m_Sxy = array_kernel[0];
            variance_Sxy = array_kernel[1];
            double k = maxValueGlobal / maxValueKernel;
            if (m_Sxy >= k0 * m_G && m_Sxy <= k1 * m_G && variance_Sxy >= k2 * variance_G && variance_Sxy <= k3 * variance_G)
            {
                outputImage.at<uchar>(row, col) *= k;
            }
        }
    }
}