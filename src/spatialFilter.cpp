/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-02-03 13:51:42
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-02-03 13:51:42
 * @Description: this file is about anthor image enhancement method, spatial
 * filter. it is a device, you can also name it used spatial filter device.
 * it is widely used in various image processing aspect. we will just consider the
 * application in image enhancement.
 * the image enhancement method that filter by changing or inhibition of the frenquency
 * component of the image. the filter device has low frequency and high frequency.
 * the application of low frequency is to smooth the image by blurred image.
 * and the spatial filter device can smooth the image directly on the image itself.
 * this is a spatial domain handle method.
 * the similarly, spatial filter handle is also change the gray value directly used 
 * the cooridinates function value or the neibohood of the center coordinates function value.
 * this changed concept is similar to the former gray vlaue transformation. they are
 * all changed the gray value directly on the image itself and the mapping value
 * is the corresponding funtion value. they are all transformation in the spatial domain.
 * of course, you can also name the filter used kernel. this kernel is defined by ourselves.
 * the kernel in local transformation is scaned matrix function, each element in it
 * is determined by the orginal vlaue. but in spatial filter device this kernel is defined
 * by ourseleves, it will multiply the scaned matrix, it means you will multioly two
 * matrix, one is scaned matrix from the original image, one is the filter device kernel
 * that we defined. it may be not fit that we have name the scaned matrix used kernel
 * in local transformation, it is because we have learned deep_learning before learning
 * the digital image processing. so we habitual defined the scaned matrix as kernel.
 * then, we have figure that the difference between scaned matrix and kernel. the former used
 * original image value, the last is defined by ourselves. and the kernel will
 * multiply with the orginal scaned matrix. so we will separate the scaned matrix and kernel
 * in last note.
 * 
 * the spatial filter device involved linear and nonlinear. 
 * the difference between local histogram transformation and spatial filter device
 * is the former just scaned the original image, and the mapping value will be calculated
 * used orginal miltiply a constant, or the mean of scaned matrix, or the center of the scaned matrix
 * mapping value that based on the equalization transformation for the scaned matrix, or
 * based on the matching transformation for the scaned matrix. so you can find the gray value transformation
 * that based on the histogram local transformation are all mapping the original value
 * based on the orginal scaned matrix indicators. but the spatial filter device
 * will add a kernel matrix that be defined by ourselves to influence the original gray value.
 * multiply the original scaned matrix by the kernel. we will get more powerful image 
 * transformation result. the convolution in deep learning is also used the concept. of course the digital
 * image processing will also use the concept of convolution.
 * 
 * what we consider is m*n size kernel, m = 2a+1, n = 2b+1, m and n are all odd number. the m is the 
 * row number in kernel, and n is the column numbers in kernel. the length of the kernel is (m-1, n-1)
 * the linear spatial filter that m*n size kernel to M*N original image can show used the follow expression
 * we should image the shape and location that the kernel corresponding to the origianl image.
 * the center coordinated of kernel is 0,0, and the origin point of the original image is 0,0
 * so you can find the relationship that the left upper scaned matrix corresponing to the kernel.
 * scaned the kernel size from the origin point, and the origin point of the kernel is the center point.
 * the original element use f(x, y) represent. and coefficient of the kernel used w(s, t) represent.
 * the size of kernel is m*n, and m = 2a+1, n = 2b+1
 * then the scaned matrix multiply the kernel is euqal to the expression as follow. x, y is each element for
 * the matrix. w(s, t) is each coefficient of the kernel. g(x, y) is the result. the x and y is fixed coordinates
 * what we want to get is the g(x, y), it is the gray value we want to mapping the f(x, y). so you will get
 * the aim about spatial filter transformation. it is find the mapping value for f(x, y) what gray value in 
 * original image about the x, y coordinates. coordinates x, y in orginal image is corresponding to the origin
 * point in kernel matrix. so just you can corresponding the w(s, t) and f(x+s, y+t)
 * and the range -a to a and -b to b menas accumulative each element of the (2a+1 * 2b+1) kernel matrix 
 * multiply each element in scaned original matrix. then change first x, y coordinates in the original image
 * used g(x, y). then ++x, y coordinates, make the center of the kernel can access each element in original image.
 * g(x, y) = Σs=-a_aΣt=-b_b w(s, t)*f(x+s, y+t)
 * why use the center of the kernel matrix as the origin point? in order to simple the expression.
 * 
 * 
 * the related calculation process as follow.
 * move the center of the kernel in the original image, and calculate the sum of each element. this is the simple
 * spatial filter kernel. then we will introduce the spatial convolution based on the spatial filter device.
 * util here, you can image the kernel used a two dimension array, it just like a table. the former problem about
 * the size and length are all dropped. because the size is equal to length if you image the picture is a table.
 * 
 * the kernel convolution and the spatial filter device is different but corresponding.
 * 
***********************************************************************/
#include "../include/spatialFilter.h"


/**
 * @Author: weiyutao
 * @Date: 2023-02-14 16:14:41
 * @Parameters: 
 * @Return: 
 * @Description: this function involved some official function about spatial filter.
 */
void officialFilterTest(Mat &inputImage, Mat &outputImage, int kernel_number) 
{
    if (kernel_number >= 3)
    {
        sys_error("kernel number is overflow");
    }
    Mat kernel;
    // define the kernel, this kernel can sharpen the image.
    // this Mat_ is a super application for Mat class. you can use it more convinence.
    if (kernel_number == 0)
    {
        kernel  = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    }
    if (kernel_number == 1)
    {
        // fuzzy kernel.
        kernel = Mat::ones(5, 5, CV_32F) / (float)(25);
    }
    filter2D(inputImage, outputImage, inputImage.depth(), kernel);
}

void officialImageMixTest(Mat &inputImage1, Mat &inputImage2, Mat &outputImage, float firstWeight) 
{
    // ensure the same size of two image
    if (inputImage1.rows != inputImage2.rows || inputImage1.cols != inputImage2.cols || inputImage1.type() != inputImage2.type())
    {
        resize(inputImage1, inputImage1, Size(300, 300));
        resize(inputImage2, inputImage2, Size(300, 300));
    }
    addWeighted(inputImage1, firstWeight, inputImage2, 1 - firstWeight, 0.0, outputImage);
}