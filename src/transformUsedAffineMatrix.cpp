/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-01-13 13:17:15
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-01-13 13:17:15
 * @Description: the general image transform involved linear and nonlinear transform
 * . the standard linear transform step is as follow.
 * this far, the operation we have done are all transform based on the original. what do you mean?
 * just like the function as follow, we will change the shape or coordinates show based on the original if we want
 * to scale, translation, shear and rotation, this is unstandard operation. just like you want to rotate one image.
 * step1: you should transform the coordinates show based on the rotation angle you want.
 *      you can do it based on the affine matrix or perspective.
 * step2: you should fill the element value used some method, just like linear interpolation.
 * that's all.
 * then, we can find the step1 operation we have done above is operation based on the original image.
 * but it will be not scientific in actual operation. just like the transformation type is a wide variety.
 * just like linear transform and nontransform. so we should operate the step1 based on the specific domain.
 * it is transform domain, just like the fourier transform the image to the frequency domain, the frequency is the
 * transform domain. so if you do the step1 based on the transform domain will be correctly.
 * then the standard step will be as follow.
 * step1: transform domain, just like fourier transform of the original image. it will transform the image from
 *          space domain to transform domain.
 * step2: transform the coordinated show and fill the transformed coordinates.
 * step3: reverse transform. it means transform the image from transform domain to space domain.
 * 
 * space domain(original input image) ---fourier transform---> tranform domain(frenquency domain)
 * ---change the cooridates and fill the value---> transform domain ---reverse transform---> space domain(output image).
 * we will change the operation idea in the last file.
 * 
***********************************************************************/
#include "../include/transformUsedAffineMatrix.h"

// they are all element operations above descriped. then we will descrip vector and matrix operation.
// this is generally used in deep learning.

// image transformation. it is the application for the coordinated operation in one image.
// then, we will implement the scale, rotation, translation and shear.
/**
 * @Author: weiyutao
 * @Date: 2023-01-10 14:52:43
 * @Parameters: 
 * @Return: 
 * @Description: 
 * geormetric operations, it can also be named rubber membrance transform.
 * step1: space transform, it means space transform to the coordinates.
 *      just like T((x, y)), notice, the input is coordinates not the element value that
 *      coordinated corresponding to. just like (x', y') = (x. y) / 2 = (x / 2, y / 2);
 *      this will result to reduce half of the original image.
 * step2: gray level interpolation. we have learned some contents for the interpolation.
 *      just like resizing one image, involved linear interpolation and binary linear 
 *      interpolation and three linear interpolation to resize the image.
 *      but we will use it in the geormetric operations.
 * affine transform. involved scaling transform, translational transform, rotation transform,
 * shear transform.
 * then, we have learned the concept about affine transform, it belong to the geormetric operations.
 * it means you want to change the size based on the original image, but you want to 
 * get an extremely similar transformation result. so you should define the size or shape according to
 * the rules you want, then you should use interpolations method to fill the element to the new shape image
 * we will use binary linear interpolation metho to do the step 2.
 * then we wii implement the image rotation case. we will use affine transformation to 
 * implement the function.
 * 
 * the matrix, you can also named Mat size. it is bigger after the rotation.
 * of course, we have many method to deal with it.
 * first, you can trim it to the same size like the original image.
 * second, you can keep the size used a bigger Mat size.
 * opposite rotation. counterclockwise
 * negative rotation. clockwise rotation.
 * arround the origin point.
 * arround the angular point.
 * 
 * then, we will learn how to calculate the coordinated transform. in order to handle this proble.
 * what rotate based on the center of the image. we should known all the step.
 * step1: conversion coordinates show from left upper original to center. because you want to rotate based on center.
 * step2: canclute the coordinated after rotating used polar coordinates.
 * step3: calculate the left border, right border, up border and down border, then get the size after rotating.
 * step4: conversion coordinates show from center to left upper original.
 * 
 * 
 * just like the matrix A as follow
 * 0,0   0,1   0,2   0,3   0,4
 * 1,0   1,1   1,2   1,3   1,4
 * 2,0   2,1   2,2   2,3   2,4
 * 
 * you should conversion the coordinates show from left upper original to center
 * -1,-2   -1,-1   -1,0   -1,1   -1,2
 * 0,-2     0,-1    0,0    0,1    0,2
 *  1,-2    1,-1    1,0    1,1    1,2
 * the transform rule is :
 * 
 * 
 * if you want to rotate this matrix based on the clockwise. it means we will rotate it -90 angle.
 * the coordinates will be as follow based on the rule
 * i' = i*cosθ - j*sinθ
 * j' = i*sinθ + j*cosθ
 * sin(-90) = -1, cos(-90) = 0
 * we can get the coordinate based on the rule above
 * 0,1 -> 1,0; 0,2 -> 2,0; 0,3->3,0; 0,4 -> 4,0
 * 1,0 -> 0,-1; 1,1 -> 1,-1
 * 0,0 -> 0,0
 * -1,0 -> 0,1
 * 1,2 -> 2,-1
 * -1,-2 -> -2,1
 * 
 * the coordinates will be B as follow after rotation based on the matrix A.
 * -2,-1  -2,0  -2,1
 * -1,-1  -1,0  -1,1
 * 0,-1   0,0    0,1
 * 1,-1   1,0   1,1
 * 2,-1   2,0   2,1
 * we can check the accuracy of the expression. it is right. but it is the use of polar coordinates method
 * to calculate?
 * 
 * the center of A is 1,2
 * the center of B is 2,-1
 * how to calculate the center of B based on A matrix? you should calculate it based on rotation angle
 * and scale rate. of course, you can also rotation a matrix used another matrix. we have rotate one matrix
 * used expression rule. another method is polar coordinated method, it need to define the affine matrix.
 * rotate matrix is a classic algorithm. we will learn it in detail after. but it is just aimed to
 * improve the efficiency of the rotation, it is not the proble at here. here we focus on how to rotate.
 * of course, you can simple to implement the specific angle rotation, just like the angle is equal to +180
 * or -180, you can transpose the matrix first, then flip it.
 * but here we want to define the function that suitable all angles.
 * 
 * then, we will compare these method.
 * method 1: it is dedicated to the rotation based on the center of image. 
 *      you can use the expression, but you should transform the left upper show to center, just like we have
 *      analysised above:
 *          i' = i*cosθ - j*sinθ
 *          j' = i*sinθ + j*cosθ
 *      because the generay show method is left upper, so you should transform first and then back again.
 *      this method is dedicated to image rotation based on the center.
 * method 2: it is generally used all transform except image translation. why, you should know all affine matrix
 *          about image transformation first. this method used a two dimension matrix as the affine matrix of each
 *          transformation. but it is not gennerally used.
 * method 3: it is generally used all transformation. you should transform the coodinates from descartes to polar coordinates.
 *          why, because we will use it to handle the image translation. we can give all affine matrix.
 * the descartes coordinates just like (x, y)
 * the polar coordinates just like (x, y, 1)
 * (x, y) == (x, y, 1)
 * polar(x, y, z)-transform->descartes(x/z, y/z), so polar will be eqaul to descartes if the third demension is 1.
 * we will use this concept to handle this problem. just like as follow.
 * 
 * identity affine matrix
 * 1    0   0
 * 0    1   0
 * 0    0   1
 * it is a unit matrix, we can use it left multiply each polar coordinated of  the original image.
 * (x, y, 1)^T, it is a vector.
 * 1    0   0
 * 0    1   0  @  (x, y, 1)^T = (x, y, 1)^T, the result is equal to the original coordinates. so it is the identity affine matrix.
 * 0    0   1
 * 
 * scale affine matrix
 * cx   0   0
 * 0    cy  0   @  (x, y, 1)^T = (cx*x, cy*y, 1)^T, the result is the scale of the original image.
 * 0    0   1
 * 
 * rotation based on the center of image
 * cosθ     -sinθ       0
 * sinθ     cosθ        0    @   (x, y, 1)^T = (cosθ*x-sinθ*y, sinθ*x+cosθ*y, 1)^T, it is equal to the expression we have learned above.
 * 0        0           1
 * 
 * translation affine matrix
 * 1    0   tx
 * 0    1   ty   @   (x, y, 1)^T = (x+tx, y+ty, 1)^T
 * 0    0   1
 * 
 * vertical shear affine matrix
 * 1    sv  0
 * 0    1   0   @  (x, y, 1)^T = (x+sv*y, y, 1)^T
 * 0    0   1
 * 
 * horizontal shear affine matrix
 * 1    0   0
 * sh   1   0   @  (x, y, 1)^T = (x, y*sh+x, 1)^T
 * 0    0   1
 * 
 * you should have noticed the difference between translation affine matrix and the other affine matrix.
 * if you have not transform the coordinates from descartes to polar, the translation affine matrix will
 * not be defined. but the other affine matrix can use 2*2 matrix to define. only the translation
 * affine matrix need to be defined used 3*3 matrix. but you must have noticed, that the third row for each 
 * affine matrix is same. it is corresponding to the result coordinates, the third dimension is always 1.
 * it is useless for us, so we can delete the third row for each affine matrix at the time of actual use.
 * so we will use a 2*3 affine matrix to implement the image transformation.
 * the affine matrix we can get it used official function getRotationMatrix2D in opencv. it will return
 * a 2*3 affine matrix, you should pass the center of your image, rotation angle and scale rate.
 * 
 * then, we will implement the rotation used the affine matrix first.
 * step1: transform coordinates from original left upper show to center.
 * step2: get the affine matrix
 * step3: calculate the offset and change the affine matrix based on the step2. calculate the size of outputImage
 * step4: affine matrix @ image matrix show used center coordinates.
 * step5: fill the value to the new matrix used linear interpolation method.
 * strp6: transform coordinates from center to original left upper show.
 * we will implement the function last.
 */
void rotationUsedAffineMatrix(Mat inputImage, Mat &outputImage, int angle, float scale) 
{
    // you should init the Mat variable first, or you will get the error.
    // outputImage = Mat::ones(inputImage.rows, inputImage.cols, CV_8UC1);
    // define a 3*3 unit matrix
    
    // notice, the affine matrix should define used float or double to store, because the element in it has sin cos.
    // notice, the first param of Size is width, the second param is height.
    Mat affineMatrix = Mat::eye(Size(3, 2), CV_64F);
    int height;
    int width;
    double width_rotate, height_rotate;
    // the first step has not implemented.
    // step2: manual define the affine matrix and calculate the offset and the size of outputImage.
    if (angle > 1)
    {
        // notice, if you want to manual define the affine matrix, you should transform the coordinates show
        // from the default left upper show to the center show. then based on it to get the affine matrix.
        // just like, the affine matrix of center @ the coordinates show based on the center = the affine matrix 
        // of original @ the coordinates show based on left upper.

        float a = cos(angle);
        float b = sin(angle);
        height = inputImage.rows;
        width = inputImage.cols;
        double radian = angle * CV_PI / 180;
        width_rotate = fabs(width * scale * cos(radian)) + fabs(height * scale * sin(radian));
        height_rotate = fabs(width * scale * sin(radian)) + fabs(height * scale * cos(radian));
        affineMatrix.at<double>(0, 0) = a;
        affineMatrix.at<double>(0, 1) = -b;
        affineMatrix.at<double>(1, 0) = b;
        affineMatrix.at<double>(1, 1) = a;
        affineMatrix.at<double>(0, 2) += (width_rotate - width) / 2;
        affineMatrix.at<double>(1, 2) += (height_rotate - height) / 2;
    }
    
    // this function involved inputImage@affineMatrix and the lienar interpolation method at least.
    // and it will set the size of outputImage used the fourth param. if it is smaller than the result
    // image inputImage@affineMatri, the image will incomplete show. so you should calculate the size.
    warpAffine(inputImage, outputImage, affineMatrix, Size(width_rotate, height_rotate), INTER_LINEAR, 0, Scalar(0));

}


// implement the scale, rotation, translation and shear used the official function that opencv provided.
// notice, we will use affine transformation to implement these function.
// the affine transformation method is affineTransform in opencv

/**
 * @Author: weiyutao
 * @Date: 2023-01-11 14:04:21
 * @Parameters: 
 * @Return: 
 * @Description: this function is dedicated to shear the image used official function in opencv
 * , notice the difference between rotation and shear, the former is rotation and do not change
 * the shape of the image, the last will distorted the image.
 * notice the difference between getAffineTransform and getRotationMatrix2D, the same about two function
 * is all get the affine matrix. a 2*3 matrix. but the former need to use two groups of coordinates.
 * you should give the original three coordinates, and give the transformation coordinates you want to get, 
 * then it will return the affine matrix based on it. the last need to the center of the image you want to rotate.
 * the rotation angle and rate scale. the first method is suitable for the image shear and you want to rotate but you 
 * do not known the angle, you just need to give the three coordinates you want. the last function need to the specific
 * angle and scale and rotation center. but we have a problem must to handle, it is the size of outputImage we must to
 * define based on the transform, or we will display the incomplete image.
 */
void transformUsedOfficialBasedOnThreePoint(Mat src, Mat &outputImage, Size2f firstPoint, Size2f secondPoint, Size2f thirdPoint) 
{
    int width = src.cols;
    int height = src.rows;
    // you should give at least three point to get the affine transformation matrix
    vector<Point2f> srcPoint;
    vector<Point2f> dstPoint;
    srcPoint.push_back(Point2f(0, 0));
    srcPoint.push_back(Point2f(0, height));
    srcPoint.push_back(Point2f(width, 0));
    double firstWidthRate = firstPoint.width;
    double firstHeightRate = firstPoint.height;
    double secondWidthRate = secondPoint.width;
    double secondHeightRate = secondPoint.height;
    double thirdWidthRate = thirdPoint.width;
    double thirdHeightRate = thirdPoint.height;
    // notice. the define about point2f is same as size function, the first param is width, the second param is height.
    dstPoint.push_back(Point2f(width * firstWidthRate, height * firstHeightRate));
    dstPoint.push_back(Point2f(width * secondWidthRate, height * secondHeightRate));
    dstPoint.push_back(Point2f(width * thirdWidthRate, height * thirdHeightRate));
    
    // but the center has changed after sheared the image. we must calculate the offset
    // and change the affineMatrix to translation the image.
    // calculate the offset, tx, ty, and calculate the size of outputimage based on the offset
    double tx = std::min({firstHeightRate, thirdHeightRate});
    double ty = std::min({firstWidthRate, secondWidthRate});
    double minWidthRate = thirdWidthRate - ty;
    double minHeightRate = secondHeightRate - tx;
    double width_transform = width * minWidthRate;
    double height_transform = height * minHeightRate;
    // transform from src to dst based on the three point we have defined.
    Mat affineMatrix = getAffineTransform(srcPoint, dstPoint);
    // change the value of affineMatrix based on the tx, ty.
    affineMatrix.at<double>(0, 2) += tx;
    affineMatrix.at<double>(1, 2) += ty;

    warpAffine(src, outputImage, affineMatrix, Size(width_transform, height_transform));
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-11 14:06:50
 * @Parameters: default value point=1, rotation based on the center of the image.
 *      point = 0, rotation based on the original point.
 * @Return: 
 * @Description: this function is dedicated to rotation a image. you can rotation the image
 * based on the original point or center point, and so on. then we will define the rotation
 * function based on the center. of course, we can scale the image based on the function wrapAffine
 * we can pass the second param used 0.
 * this function is dedicated to rotate based on center and angle. if you want to rotate based on the 
 * left upper original, you should define another function. In addition, this function used official
 * function, we have implemented the efficient, but we have not learned the concept of it.
 * so we will learn it subsequent. because we have not found the method to calculate the image size
 * after rotation and scale. so we deleted the scale and original point param. you should recall the 
 * resize function or the resize function defined by myself to handle these problem.
 */
void transformUsedOfficialBasedOnSpecificParam(Mat src, Mat &outputImage, int angle, float scale) 
{
    Point2f pointCenter;
    int height = src.rows;
    int width = src.cols;
    // fabs function can return the abs of a float
    // radian = angle * pi(3.1415926) / 180
    // radian = l / r, r is the radius of circle. l is the arc length.
    // you should consider the scale at here.
    double radian = angle * CV_PI / 180; // converting angle to radian.
    double width_rotate = fabs(width * scale * cos(radian)) + fabs(height * scale * sin(radian));
    double height_rotate = fabs(width * scale * sin(radian)) + fabs(height * scale * cos(radian));

    // give the center point used Point class
    // we will calculate the outputImage shape in this if instruction.
    // because the different condition will be the different shape.
    // of course, if you rotation one image based on the center, then you will not move the center.
    // so we should calculate the outputImage size after rotating.
    pointCenter = Point2f((float)width / 2.0, (float)height / 2.0);
    // then we will use the finction getRotationMatrix to get the matrix after roationing
    // based on the param you given.
    // the first param is the point what you will rotation based on it.
    // the second param is rotation angle that you want.
    // the third param is the rate of scale.
    // this code will return the rotation matrix
    
    Mat rotationMatrix = getRotationMatrix2D(pointCenter, angle, scale);
    
    // after getting the rotation matrix, you should calculate the size of the rotationMatrix.
    // get the offset, because before you used warpAffine function, the new image you want to generate
    // based on the rotationMatrix, it is correctly, but the outputImage will not suitable the new transformation
    // image. so you should add the translation param. it just like the new affine matrix as follow
    // rotation affine matrix based on center.
    /*  
    cosθ    -sinθ   0
    sinθ    cosθ    0
    0       0       1 

    but we want to show the complete new image in old outputImage. so we should translation the new matrix.
    the translation matrix
    1    0   tx
    0    1   ty
    0    0   1  

    so we define the new affine matrix that involved rotation and translation function
    cosθ    -sinθ   tx
    sinθ    cosθ    ty
    tx = width_rotate - width / 2
    ty = (height_rotate - height) / 2
    then we will display complete image in the old outputImage.
    so we have used rotation affine matrix and translation affine matrix.
    so this function getRotationMatrix2D we can drop, because it is the function that get affine matrix,
    we can manually define the affine matrix.
    */
    rotationMatrix.at<double>(0, 2) += (width_rotate - width) / 2;
    rotationMatrix.at<double>(1, 2) += (height_rotate - height) / 2;

    // notice, the first param is the original image
    // the second param is the output image
    // the third param is the image shape you want to change.
    // the fourth param is the outputimage size you want to show.
    // notice, the difference between the third param and the fourth param.
    // the former is the element value range, the last is the window image size.
    // just like the outputimage shape is 100*100
    // but the rotationMMatrix is the matrix coodinates you want to transform the value from
    // original image to the new image rotationMatrix.
    // then, if you do not calculate the outputImage shape, you will get a incomplete image display.
    // then, we will calculate the display image shape, it can be also named the size of outputimage.
    // how to calculate?
    // a matrxi rotation one angle, how much the size of a new matrix can contain the rotationMatrix, what
    // the image after rotationing.
    if (src.channels() == 1)
    {
        warpAffine(src, outputImage, rotationMatrix, Size(width_rotate, height_rotate), INTER_LINEAR, 0, Scalar(0));
    }
    warpAffine(src, outputImage, rotationMatrix, Size(width_rotate, height_rotate), INTER_LINEAR, 0, Scalar(255, 255, 255));
}

// then, we have the simple image rotation method, but it can just rotation 90.
// this method used tranpose method, and flip method.
// clockwise = 1, clockwise rotating.
// clockwise = 0, counterclockwise rotating.
void imageRotationSimple(Mat &inputImage, bool clockwise = 1) 
{
    transpose(inputImage, inputImage);
    // you can change the third param to get the different efficient.
    // the third param can be 0 or 1, corresponding mirror based on the x axis or y axis.
    // then, combine the former operation transpose the matrix. 
    // if the third param value is 0, it means counterclockwise rotating. otherwise, it will be clockwise rotating.
    flip(inputImage, inputImage, clockwise);
}