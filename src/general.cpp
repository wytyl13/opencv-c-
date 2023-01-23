#include "../include/general.h"

void sys_error(const char *str) 
{
    perror(str);
    exit(1);
}

// then, we will learn element operations
// we defined a simple function, that T(z) = -z + 255
void elementOperation(Mat inputImage, Mat &outputImage) 
{
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    outputImage.create(inputImage.size(), inputImage.type());
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            outputImage.ptr<uchar>(i)[j] = -1 * inputImage.ptr<uchar>(i)[j];
            outputImage.ptr<uchar>(i)[j] += 255;
        }
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-10 12:09:46
 * @Parameters: 
 * @Return: 
 * @Description: show many image in one window, it means you should define the param
 * that you want to used show type, just like, you should define the min or max width and
 * height if you show different number image. we defined the different min and max width and height
 * when you pass the different numbers image.
 * the max numbers image you can show is 12
 * 1-4: 
 *      1 360*600   1,1
 *      2 360*600   1,2
 *      3 360*600   2,2
 *      4 360*600   2,2
 * 5-9:
 *      5 180*300    2*3
 *      6 180*300    2*3
 *      7 180*300    3*3
 *      8 180*300    3*3
 *      9 180*300    3*3
 * 10-12, the max width and height is 150px
 *      10 90*150    4*3
 *      11 90*150    4*3
 *      12 90*150    4*3
 * just like, if one picture is 200*600, if the max width and height is 300.
 * the you should scale the size of image to 100*300. the height and width can not more than the max height and width.
 * 
 * then, we have define the fixed parma that the function display multi image.
 * if you want to show multi image, you will get the fixed size based on the numbers of the input imagee.
 */
void imshowMulti(string &str, vector<Mat> vectorImage) 
{
    int numImage = (int)vectorImage.size();
    int w, h; // w means the image numbers for row, h means the image numbers for columns.
    // just like, w is 2 if you want to show two image in one window.
    // w is 1 if you want to show one image in one window.
    int height, width; // the height, width that each image based on the numbers of input image.

    if (numImage <= 0)
    {
        printf("the image numbers arguments you passed too small!");
        return;
    }
    else if (numImage > 12)
    {
        printf("the image number arguments you passed too large!");
        return;
    }

    if (numImage <= 4)
    {
        height = 360; width = 600;
        switch (numImage)
        {
        case 1:
            h = 1; w = 1;
            break;
        case 2:
            h = 1; w = 2;
            break;
        default:
            h = 2; w = 2;
            break;
        }
    }
    else if (numImage >= 5 && numImage <= 9)
    {
        height = 180; width = 300;
        switch (numImage)
        {
        case 5:
            h = 2; w = 3;
            break;
        case 6:
            h = 2; w = 3;
            break;
        default:
            h = 3; w = 3;
            break;
        }
    }
    else
    {
        height = 90; width = 150;
        h = 4; w = 3;
    }

    Mat dstImage = Mat::zeros(60 + height * h, 90 + width * w, CV_8UC1);
    // notice, you should start from 20,20. because you should reserved space between two image.
    // m, n is cooresponding the element corrdinates x, y in the dstImage.
    // this bigImage is the output image than involved all input image.
    for (int i = 0, m = 20, n = 10; i < numImage; i++, m += (10 + width))
    {
        if (i % w == 0 && m != 20)
        {
            // if true, you should start from 20, because it must be the right of the window.
            m = 20;
            n += 10 + height;
        }
        // frame of a region in original image dstImage.
        // this region used variable imgROI to show.
        Mat imgROI = dstImage(Rect(m, n, width, height));
        // notice. the first param of Size is width, the second param is height.
        resize(vectorImage[i], imgROI, Size(width, height));
    }
    imshow(str, dstImage);
}

void printMap(const map<int, Mat, compareMap> &mapBitPlane)
{
    for (map<int, Mat, compareMap>::const_iterator it = mapBitPlane.begin(); it != mapBitPlane.end(); it++)
    {
        cout << "key = " << it->first << "value = " << it->second.size() << endl;
    }
}


/**
 * @Author: weiyutao
 * @Date: 2023-01-19 21:39:18
 * @Parameters: 
 * @Return: 
 * @Description: draw one line.
 */
void drawLines(Mat &inputImage, Point one, Point two) 
{
    line(inputImage, one, two, Scalar(255), 3, 0);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-19 22:01:33
 * @Parameters: 
 * @Return: 
 * @Description: draw one polygon based on the vectorPoints.
 * you should ensure each points is next. or you will get the error lines.
 */
void drawPolygon(Mat &inputImage, vector<Point> vectorPoints) 
{
    polylines(inputImage, vectorPoints, true, Scalar(255), 2, 8);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-20 08:32:55
 * @Parameters: 
 * @Return: 
 * @Description: you can only screenShots the recangular region.
 * we used Rect object to passrecangular region.
 */
void screenShots(Mat inputImage, Mat &outputImage, Rect rect) 
{
    // it is very simple, you just need to define the Mat object used the Rect
    // this is a construcure method for Mat object.
    outputImage = inputImage(rect);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-20 08:49:24
 * @Parameters: 
 * @Return: 
 * @Description: cut image based on multiple points. remain the region that you interested in, 
 * set the other region used 0.
 */
void cutImage(Mat inputImage, Mat &outputImage, vector<Point> vectorPoints) 
{
    // first, you should define a mat mask.
    Mat mask = Mat::zeros(inputImage.size(), CV_8UC1);    
    fillPoly(mask, vectorPoints, Scalar(255));
    cout << (int)mask.ptr<uchar>(1)[1] << endl;
    // then, bitwise inputimage and mask.
    // of course, you can also use copyTo function.
    // the first param in copyTo function is src image, the second param is mask
    // Mat, the element value of region you interested in set used 255, the other
    // region element value set used 0.
    // of course, the function of copyTo can also has a extra efficient.
    // that's you can copy the image to a region, this region can be any 
    // Mat object. but you should notice, the size of these two image must be same.
    bitwise_and(inputImage, mask, outputImage);
}
