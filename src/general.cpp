#include "../include/general.h"

#define MAX_COLUMN = 7



// overload the + and += operators. you can use string+int or
// string+=int.
string operator+(string &content, int number)
{
    string temp = "";
    char t = 0;
    while (true)
    {
        t = number % 10 + '0';
        temp = t + temp;
        number /= 10;
        if (number == 0)
        {
            return content + temp;
        }
    }
}

string& operator+=(string &content, int number)
{
    return content = content + number;
}




/**
 * @Author: weiyutao
 * @Date: 2023-02-15 11:43:36
 * @Parameters: pts, coordinates set, mode, the attribution mode, you can use the macro.
 * @Return: any type pointer, you can receive double and vector or other. you can calculate
 * the length, area and center of the feature points set.
 * @Description: this function will calculate all attribubtion based on the coordinates.
 * you should pass one coordinates set, the type is vector here, of course, you can define
 * used other type. just like ACS::LENGTH.
 * 
 * 00 01 02 03 04 05 06 07
 * 10 11 12 12 14 15 16 17
 * 20 21 22 23 24 25 26 27
 * 30 31 32 33 34 35 36 37
 * 40 41 42 43 44 45 46 47
 * 50 51 52 53 54 55 56 57
 * 60 61 62 63 64 65 66 67
 * 70 71 72 73 74 75 76 77
 * 
 * rect(1, 2, 4, 4), it means define an rectangular the original is (1,2),
 * width is 5, height is 5. it means from 1,2 to 5,6. then how to get the
 * center of the rectangular? the center is 3,4.
 * the center is equal to the original add the height/2, width/2.
 * 
 * but it is not suitable for the irregular object. but you can draw the minimize rectangular
 * for the irregular object. then it will be a though to get the center from a 
 * feature points set of one irregular object. then, we should transform the points set
 * from vector<Point> to mat or other data type.
 * 
 * you'd better not return a pointer, because you should define a pointer in the current
 * function first, but it is worth to notice that you need not always to malloc a memory.
 * it means you need not always to define the pointer, you can statement a pointer variable 
 * that point to an exist memory in heap. just like you have malloc a memory in this function.
 * and you have return the pointer variable that point to there, then you should statement
 * a pointer variable in the function what you used this return pointer function.
 * but you should notice you should free the pointer when it is usefuless. 
 * you can define a pointer like as follow two method.
 * int *pointer1 = (int *)malloc(sizeof(int) * 10);
 * int *pointer2 = pointer1, pointer1 is an exists pointer in heap.
 * then, you can free pointer2, it is equal to free the memory created in heap.
 * because the pointer1 and pointer2 are both the pointer variable, they are all
 * stored the address in the same address that has malloced in heap when you statemented pointer1.
 * we have tested the center point is the mean of x and y in the Mat what is 
 * another matrix data type based on all the feature points set of one object in one image.
 * 
 * so we can conclude that the mean of x and y of the object feature points Mat is the center point
 * if you want to get the center point based on a irreugular shape object.
 */
void *calculateAttributionBasedOnFeaturePoints(vector<Point> &pts, int mode) 
{
    double *returnPointer;
    int size = pts.size();
    // each row can mean a coordinate.
    Mat ptsCoordinatesMat = Mat(size, 2, CV_64FC1);
    for (int i = 0; i < size; i++)
    {
        ptsCoordinatesMat.at<double>(i, 0) = pts[i].x;
        ptsCoordinatesMat.at<double>(i, 1) = pts[i].y;
    }
    // but we have tested the method to calculate the center point based on
    // the points set, it is the mean about all x and y. not the method above.
    // so this method will be simple, yuo just need to get the mean of x and y in mat.
    /* // you should get min row coordinate and column coordinate.
    double minCoordinateX, maxCoordinateX, minCoordinateY, maxCoordinateY;
    minMaxLoc(ptsCoordinatesMat.colRange(0, 1), &minCoordinateX, &maxCoordinateX, 0, 0);
    minMaxLoc(ptsCoordinatesMat.colRange(1, 2), &minCoordinateY, &maxCoordinateY, 0, 0);

    // you should calculate the center point based on these four number.
    // the original is minCoordinateX, minCoordinateY, the height = maxCoordinateY - minCoordinateY
    // the width is equal to maxCoordinateX - minCoordinateX
    // the center point = minCoordinateX + width / 2, minCoordinateY + height / 2;
    double *list = (double *)malloc(sizeof(double) * 4);
    double width = maxCoordinateX - minCoordinateX;
    double height = maxCoordinateY - minCoordinateY;
    list[0] = minCoordinateX + width / 2;
    list[1] = maxCoordinateY + height / 2; */


    if(mode == ACS::CENTER)
    {
        returnPointer = (double *)malloc(sizeof(double) * 2);
        returnPointer[0] = cv::mean(ptsCoordinatesMat.colRange(0, 1))[0];
        returnPointer[1] = cv::mean(ptsCoordinatesMat.colRange(1, 2))[0];
    }

    return returnPointer;
}

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

void printOneArrayPointer(const double *array, int length = 256) 
{
    for (int i = 0; i < length; i++)
    {
        cout << array[i] << endl;
    }
}


void printTwoArrayPointer(const double *array1, const double *array2, int length = 256) 
{
    for (int i = 0; i < length; i++)
    {
        cout << array1[i] << " " << array2[i] << endl;
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-31 09:55:51
 * @Parameters: 
 * @Return: 
 * @Description: you can use this function show multi arrayList what shape is 256*1.
 * if you pass two parameters, it will return the 256*2 information.
 * becasue the pointer of the array can not calculate the size, so we do not 
 * judge the size of the array you have passed. of course, you can also use the class to encapsulation
 * these print method. but you can also define these function used the uncertain parameters method.
 */

void printArrayListPointer(double *array, ...)
{
    va_list arg;
    va_start(arg, array);
    // you can use memcpy function to append two 256*1 dimension array to a 256*2 dimension array.
    double argArray[7][256];
    double *argValue;
    memcpy(argArray[0], array, 256 * sizeof(double));
    int i = 1;
    do
    {
        
        argValue = va_arg(arg, double *);
        memcpy(argArray[i], argValue, 256 * sizeof(double));
        i++;
        if (i == 7)
        {
            sys_error("the parameters has overflow");
        }
    } while (argValue != 0);
    va_end(arg);
    for (int n = 0; n < 256; n++)
    {
        for (int m = 0; m < i; m++)
        {
            cout << argArray[m][n] << " ";
        }
        cout << endl;
    }
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-01 09:28:45
 * @Parameters: 
 * @Return: 
 * @Description: define a thread function to handle the localHistogram transformation
 * problem, because its amount of calculation is large.
 */
void thread_function(int i)
{
    cout << "子线程" << i << "开始执行" << endl;
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

void freePointer(void *pointer) 
{
    free(pointer);
    pointer = NULL;
    if (pointer != NULL)
    {
        free(pointer);
    }
}