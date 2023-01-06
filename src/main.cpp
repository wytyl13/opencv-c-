#include <iostream>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>

#include <time.h>
#include <vector>

using namespace cv;
using namespace std;

typedef unsigned char uchar;
// we will define some function about image zoom used c++, 
// involved linear interpolation, binary linear interpolation, three linear interpolation.
// we will define the method for gray image first. the method as follow is not suitable for
// color images.
// first, you should define a function that can get the element about one coordinates. it means 
// give x and y, this function should return the corresponding element.
// the grayscale is range from 0 to 255, so we should define the unsigned char to accept each element in picture.
// unsigned char type can store the number that range from 0 to 255.

uchar get_scale_value(Mat &input_image, int x, int y) 
{
    uchar *p = input_image.ptr<uchar>(x);
    return p[y];
}

// then we should define the image zoom function, we will define a simple method that is named linear interpolation.
// any +-*/ float/double = double
// int +-*/ float = double
// int +-*/ double = double
// char +-*/ int = int
// int / int = int
// notice, you should ensure the grayscale of input_image is range from 0 to 255, it means the gray image downloaded
// from network is not ensure it. so you shoud use cvtColor function to get the gray image.
// float/double  modify  int = double. you can use any data type to accept the result, because the result will
// implict type conversion.
Mat scale(Mat &input_image, int height, int width) 
{
    // CV_8UC1 means gray image, 8 bits and single channel
    // 8 means 8 bits, uc means unsigned char, 1 means 1 channel. it means we will create a gray picture.
    Mat output_image(height, width, CV_8UC1);
    output_image.setTo(0); // init all element used 0
    float scale_rate_h = (float)input_image.rows / height;// calculate the height rate
    float scale_rate_w = (float)input_image.cols / width;// calcute the width rate
    // traverse each element used traditional method for circle.
    for (int i = 0; i < height; i++)
    {
        uchar *p = output_image.ptr<uchar>(i);
        for (int j = 0; j < width; j++)
        {
            // you should calculate the coordinates of the adjacent based on x, y and rate 
            // during the period of traversing.
            // of course the result will be float, but we want to get a integer, so we should casts
            // from float to int. take down the whole will happen after casts from float to int.
            // float * int = int;
            int scale_i = scale_rate_h * i;
            int scale_j = scale_rate_w * j;
            p[j] = get_scale_value(input_image, scale_i, scale_j);
        }
    }
    return output_image;
}


// of course, we can define the scale function used binary linear interpolation method.
// this method just like the follow.
/*
-->
|

i, j                   i+1, j       
       i+u, j+v

i, j+1                 i+1, j+1
0 < u, v < 1.
because the i+u, j+v what elemet we want to calculate not neccessarily at the center of the image array.
but we must could find the latest four elements. then we can calculate the result based the four elments.
f(i+u, j+v) = (1-u)*(1-v)*f(i, j) + (1-u)*v*f(i,j+1) + u*(1-v)*f(i+1,j) + u*v*f(i+1,j+1)
the f(i+u,j+v) is the result we want to get. then we start to define this function.
notice, input corrdinates are not the int. but is the float.
*/
uchar get_scale_value_binary(Mat &input_image, float _i, float _j) 
{
    int i = _i;
    int j = _j;
    float u = _i - i;
    float v = _j - j;

    // handle the border problem
    if ((i + 1 >= input_image.rows) || (j + 1 >= input_image.cols))
    {
        uchar *p = input_image.ptr<uchar>(i);
        return p[j];
    }
    uchar *p = input_image.ptr<uchar>(i);
    uchar x1 = p[j]; // f(i, j)
    uchar x2 = p[j + 1]; // f(i, j + 1);
    p = input_image.ptr<uchar>(i + 1);
    uchar x3 = p[j]; // f(i+1, j)
    uchar x4 = p[j + 1]; // f(i+1, j+1)
    return (1 - u) * (1 - v) * x1 + (1 - u) * v * x2 + u * (1 - v) * x3 + u * v * x4;
}

// then we will define the function that generate the image after interpolating.
Mat binary_linear_scale(Mat &input_image, int height, int width) 
{
    Mat output_image(height, width, CV_8UC1);
    output_image.setTo(0); // init all element used 0
    float scale_rate_h = (float)input_image.rows / height;// calculate the height rate
    float scale_rate_w = (float)input_image.cols / width;// calcute the width rate
    for (int i = 0; i < height; i++)
    {
        uchar *p = output_image.ptr<uchar>(i);
        for (int j = 0; j < width; j++)
        {
            // you should calculate the coordinates of the adjacent based on x, y and rate 
            // during the period of traversing.
            // of course the result will be float, but we want to get a integer, so we should casts
            // from float to int. take down the whole will happen after casts from float to int.
            // float * int = int;
            float scale_i = scale_rate_h * i;
            float scale_j = scale_rate_w * j;
            p[j] = get_scale_value_binary(input_image, scale_i, scale_j);
        }
    }
    return output_image;
}


// then we will learn the three linear interpolation. it means we should get the nearest coordinates based on 16 near coordinates.
// binary linear interpolation is based on four near coordinates.
// you should known that one image array is start from 0, 0 where location at the upper left corner of the image array. 
// it is a Mat variable in opencv. so the eight near coordinates is just like as follow.
/*
i,j                i,j+1
       i+u,j+v

i+1,j              i+1,j+1
it is four near coodinates above. then we should show 16 near coordinated based on it.
the difference between binary linear interpolation and three linear interpolation is the former is dedicated to an array, 
and the last is dedicated to a three dimension. just like the information above, if you add the third dimension for it.
you can find the extra 12 near coordinates. then you will have eight near coordinates. just like the information as follow

    
            i,j,z                      i,j+1,z

                i+u,j+v,z+x


    i+1,j,z                     i+1,j+1,z

but the information about three linear interpolation is not easy to show. so we will give up to show it in text file.



then we should show the information about pathways what is the shortest route from element p to q based on the adjacency and V{2, 3, 4}.
just like the information as follow. the pathways must through based on the rule about adjacency.
3   4   1   2   0       a   b   c   d   e
0   1   0   4   2       f   g   h   i   q
2   2   3   1   4       k   l   m   n   o
3   0   4   2   1       p   z   r   s   t
1   2   0   3   4       u   v   w   x   y
p(3,0) = 3 --> q(1,4) = 2
the shortest four pathways based on four adjacency, the each element that pathways through must be the four adjacency for p and q. and 
q is the four adjacency of p. and you can just up or down if you used four adjacency.
    p(3,0) = 3 -up-> 2 -right-> 2 -right-> 3 -down-> 4 -right-> 2 -down-> 3 -right-> 4=array(4,4)
    this pathways is can not arrive q, so the four adjacency is not exists for this array image.

the shortest eight pathways based on eight adjacency, you can up, down, upper right, upper left, lower right and lower left, but the rule
must be suitable for the eight adjacency. you can select the shortest route from up, down, left, right and upper right, upper left, lower
right, lower left.
    p(3,0) = 3 -upper right-> 2 -right-> 3 -upper right-> 4 -right-> 2 = q(1,4)
    the shortest route is four, it means the eight pathways is four.

then, we will consider m pathways, we should known what is the m adjacency. just like the information as follow.

V{1}

0   1   1                   a   b   c
0   1   0                   d   e   f           
0   0   1                   g   h   i
the original image          


the center elemnt is p(1,1) = 1;
all four adjacency element for p(1,1) is array(0,1)=1
all eight adjacency element for p(1,1) is array(0,1)=1, array(2,0)=1, array(2,2)=1
all m adjacency element for p(1,1) is array(0,1)=1, array(2,2)=1. notice, we should consider the N4(p)∩N4(q)∉V
just like N4(e) = bdhf, the ND(e) = acgi, the N8(e) = N4(e) + Nd(e) = ABCDFGHI, these are neighborhood.
if you consider the four adjacency or eight adjacency. you should add the condition that the element should belong the V. if it is not
meet, you should delete the element as the adjacency of e.
if you consider the m adjacency. you should add the extra condition N4(each element for N4(e) or ND(e))∩N4(e)∉V 
just like the case above.
N4(e) = bdhf, the ND(e) = acgi, the N8(e) = N4(e) + Nd(e) = ABCDFGHI
the four adjacency of e is b, because dhf∉V{1}
the eight adjacency of e is bci
the m adjacency of e is bi, because N4(c)∩N4(e) = bf, and b=1∈V, N4(b)∩N4(e) = 0∉V， and N4(i)∩N4(e)=fh, and fh∉V
you should have known the difference between eight adjacency and m adjacency, the last is more strict and it can handle the ambiguity of the eight adjacency.


then we should give the m pathways based on the former case.
our pupose is from p to q, and give the shortest route based on the four adjacency, eight adjacency and m adjacency.
N4(p)=kz    
four adjacency of p is k    
N4(k)=fl
four adjacency of k is l
N4(l)=kgmz
four adjacency of l is km, k is ignored
N4(m)=hlrn
four adjacency of m is lr, l is ignored
N4(r) = mzws
four adjacency of r is ms, m is ignored
N4(s) = nrxt
four adjacency of s is rx, r is ignored
N4(x) = swy
four adjacency of x is sy, s is ignored.
N4(y) = xt
four adjacency of y is x, x is ignored.
util here, the route is closed, so the four pathways from p to q for this image is not exists.


then we can consider the eight pathways.
N8(p) = klzvu, eight adjacency of p is klv. v is ignored, we can choose k or l as the next step. becaused of the shortest route, we should choose l.
N8(l) = fghmrzpk, eight adjacency of l is mrpk, rpk is ignored because of the shortest route is needed.
N8(m) = ghinsrzk, eight adjacency of m is isrl, srl is ignored because of the shortest route is needed.
N8(i) = cdeqonmh, eight adjacency of i is dqom, dom is ignored because of the shortest route is needed.
so the eight pathways is from p>>l>>m>>i>>q, the length is equal to four.

then we can consider the m pathways based on the eight pathways.
N8(p) = klzvu, eight adjacency of p is klv. the m adjacency of p is k.we should add the extra condition, N4(p)∩N4(k)=0∉V, N4(p)∩N4(l)=kz, and k=2∈V, N4(p)∩N4(v)=zu∉V,
    the l is not suitable for the rule of m adjacency. the v is ignored because of the shrtest route is needed, so you can just step to k.
N8(k) = fglzp, eight adjacency of k is lp, the m adjacency of k is l.N4(k)∩N4(l)=0∉V, and p is ignored. so you can just step to l.
N8(l) = fghmrzpk, eight adjacency of l is mrpk. the m adjacency of l is m. pk is ignored because it will be return. r is ignored because of the shortest route even if we have not
    judge if it is suitable for the rule of m adjacency. the m is be left.N4(l)∩N4(m)=0∉V, so you can just step to m.
N8(m) = ghinsrzk, eight adjacency of m is isrl, the m adjacency of m is i, srl is ignored because it will be return, we just need to adjust
    the suitablely of i for the m adjacency. N4(i)∩N(m)=hn∉V. so you can just step to i.
the last step, you should step to q directly.
then, so the m adjacency route from p to q is p>>k>>l>>m>>i>>q, the length is equal to five, it is bigger than the route length of eight pathways.

connectivity.
if p and q is belonged to s what an element subset of an image array. if exist a pathways that from p to q can connect all element in s.
we can conclude that p and q is connected in s. notice, this saying just applied to s set.
the connectivity can be divided into three type, just like 4-connectivity, 8-connectivity and m-connectivity based on the pathways method above.
connected component, the element set that can connect to p in s is named the connected component of p. just like it may have multiple
sets that can connect to p in s. so the elements in each pathways is a set what is named the connected component of p.
if it just has one connected component in s, then we can name s as connected set. it means a connected set just has one connected component.

regions
if r is a connected set, then r is a region of image. the defination of region is just based on four adjacency or eight adjacency. not the m adjacency.
if two regions r1 and r2. if r1 union r2 is a new connected set, so it can be named r1 and r2 is adjacency. it is the adjacency about regions.
notice, the region just involved four adjacency or eight adjacency, not has the m adjacency, it is the difference between region adjacency
and element adjacency.
*/



/**
 * @Author: weiyutao
 * @Date: 2022-10-17 15:07:48
 * @Parameters: imgDil : the dilate image; img : the original image.
 * @Return: null
 * @Description: the method to get the contours.
 */
void getContours(Mat imgDil, Mat img)
{
    // define the contours used vector. we just need to define the data type.
    // this variable contours will be a outgoing param
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // when you run the function fifnContours, the all contours data found will be given contours.
    // we can use it to do something.
    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // after finding it,  we can draw the contours in your picture based on the contours.
    // the first param is the img original, what is the picture we find contours on it.
    // the second param is the contours we have found in the previouus code function.
    // the third param means what contours you want to draw, -1 mean all the contours.
    // the fourth param is the color you draw, the fifth param is the border you drawed.
    // notice, the third of drawContours function is the index of the contours. if -1 you passed, it means
    // you want to draw all contours on the original picture.
    // drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

    // then you have get all the contours data point, so you can do something based on it.
    // just like you can draw something based on it, you can also filter the not neccesary contours based on some rule
    // you defined.
    int i = 0;
    int size_contours = contours.size();
    // a string type to store the contour shape
    string contour_shape;

    //define a vector to store the data. the size is equal to the size of contours we have found.
    // we can get the data array based on the angle not the edge.
    // we should define the conpoly to store the angles based on the contours.
    vector<vector<Point>> conPoly(size_contours);

    // define a vector that store the rectangular, we can draw the max rectangular boundary based on the contours
    // in your original picture.
    vector<Rect> boundRect(size_contours);

    // you can use for loop get each element. but this is not a good habit, you should use iterator or algorithm
    // for_each to get each element. then we can
    for (vector<vector<Point>>::iterator it = contours.begin(); it != contours.end(); it++)
    {
        // we can filter the contours based on the area of contours. we can use contourArea to get
        // the area for each contours. we can also use the for loop, beacuse we should use the index i
        // as the third param of drawContours function, but if not use it, we can alse define the index i
        // out of the for.
        int area = contourArea(*it);

        // then let us do some special thing, we have drawed the contours based on the imgDil what is the 
        // border imgCanny, then we must think if we can draw something not based on the edge found.
        // what means we can draw the line not based on the edge strictly, but based on the angle. so it is
        // we can find four angle in a rectangular, and three angle in a triangle, and many angle in a circle.
        // so we can judge the shape based on the angle numbers we have found.
        // have defined at the row 45, goto check. beacause we should define it out of the for loop.


        // we can also graph based on the boundary about the contours we have detected, the boundary is 
        // different from edge, it is a min area rectangular to show the detected edge or contours, and
        // the edge is equal to contours generally. and the graph method involed graph based on the edge and
        // graph based on the angle, if you have a irregular shape, you will get many angles. and if you have
        // a regular shape, just like rectangular, you will have four angles.

        // we should define the data variable to store the boundary based on the contours
        // have defined at the row 47, goto check. beacause we should define it out of the for loop.




        if (area > 300 && area < 500)
        {
            // if you only want to draw line based on the contours. you just need to code follow the next code.
            // drawContours(img, contours, i, Scalar(255, 0, 255), 2);

            // draw the line based on the conpoly what is the angle data pointer generate from the contours.
            float peri = arcLength(*it, true);
            // get the angle point from the contours, what is conpoly here, is the point for each contours
            // we have detected from the picture, it is found by the angles. it means if you a contours is
            // irregular, it will have more angles.
            approxPolyDP(*it, conPoly[i], 0.02 * peri, true);
            drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);

            // if you want to draw something based on the boundary, you can do it follow the next.
            // get the boundary from the conploy what is the vector stored the angle point of each contour.
            boundRect[i] = boundingRect(conPoly[i]);
            // you should give the topleft angle and bottomright angle to draw the rectangle boundary.
            // notice we graph the line in original image. this function we can apply for the face detection.
            rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2);

            // then we can do some judge based on the angle numbers for each contours we have found.
            // and we can also graph the text into the coordinate based on the boundRect, what is max boundary
            // for each contour we got it from conpoly, and the conpoly is waht we got it based on the contours.
            // this all process is from numerous to jane, it is a process of statute.
            int angle_number = (int)conPoly[i].size();
            if (angle_number == 2) { contour_shape = "line"; }
            if (angle_number == 3) { contour_shape = "triangle"; }
            if (angle_number == 4)
            {
                // we can do judgement for future. we can judge the rectangle just means what the contour 
                // has four angles, we can get it is a square or a rectangle.
                float aspRatio = (float)boundRect[i].width / (float)boundRect[i].height;
                if (aspRatio > 0.95 && aspRatio < 1.05)
                {
                    contour_shape = "squre";
                }
                else
                {
                    contour_shape = "rectangular";
                }
            }

            if (angle_number > 4) { contour_shape = "circle or irregular"; }
            // we can use .x or .y get the each coordinate. 
            // and if it is a rectangular, we can also use .width or .height to get the weight or height of the rectangular.
            // if it is not the rectangular, we can not use the .width or .height.
            // beacuse the data type are all rectangle, so we can use width and height here.
            // but the .x and .y is different from the width and height, the former is a point px in this image,
            // the last is a length for width and height. so we should known it.
            putText(img, contour_shape, {boundRect[i].x, boundRect[i].y - 5}, FONT_HERSHEY_DUPLEX, 0.75, Scalar(0, 255, 0));
            cout << conPoly[i].size() << endl;
        }
        i++;
    }
}

/**
 * @Author: weiyutao
 * @Date: 2022-10-17 22:37:44
 * @Parameters: 
 * @Return: 
 * @Description: get the plate
 */
void detectPlates() 
{
    // this function we need to the camera.
    VideoCapture capture(0);
    Mat img;
    CascadeClassifier plateCascade;
    // you should use the file haarcascade_russian_plate_number.xml
    plateCascade.load("resources/haarcascade_russian_plate_number.xml");
    if (plateCascade.empty())
        cout << "xml file not loaded" << endl;
    vector<Rect> plates;
    while (true)
    {
        capture.read(img);
        plateCascade.detectMultiScale(img, plates, 1.1, 10);
        for (long long unsigned int i = 0; i < plates.size(); i++)
        {
            // we should save the plates based on the vetor plates.
            // this plates is a vector that have four points to show a rectangular, 
            // pass four points to a img we can get a image after the cut based on the
            // four points.
            Mat imgCrop = img(plates[i]);
            // imshow(to_string(i), imgCrop);
            time_t t;
            imwrite("resources/plates/" + to_string(time(&t)) + "_" + to_string(i) + ".png", imgCrop);
            rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);
        }
        imshow("image", img);
        waitKey(1);
    }
}




int main(int argc, char const *argv[])
{
    // -----------read a picture file -----------------
    // this is read and show one picture using opencv.
    // notice: this ./ path is the current program opencv.
    // string path = "resources/image.jpg";
    // Mat img = imread(path);
    // imshow("the image", img);
    // waitKey(0);


    // ------- read a mp4 file. ----------------
    // string path = "resources/2.mp4";
    // VideoCapture cap(path);
    // Mat img;
    // if (!cap.isOpened())
    //     return -1;
    // while (cap.read(img))
    // {
    //     // read the picture and store the result as img variable.
    //     imshow(" movie", img);
    //     // the unit is milliseconds. the next code meaning is 1 millisecond interval beatween each frame
    //     // if it is zero, it will be wait forever.
    //     waitKey(20);
    // }

    // open the computer camera, you just need to set the path using a id, this id is the
    //camera in your computer. if you just have one camera in your computer, you should use zero.
    // VideoCapture capture(0);
    // Mat img;
    // while (capture.read(img))
    // {
    //     imshow("the camera", img);
    //     waitKey(1);
    // }
    
    /** then we will define some basic operation in opencv.
    // first, we read the gray picture. the level is range from 0 to 255, there
    // are 256 levels in it. the 0 is blank, the 255 is white, the 1-254 is gray.
    string path = "resources/image8.jpg";
    Mat img = imread(path);

    

    // cast the image from BGR to GRAY, and use the imgGray to store the result.
    // notice: the img and imgGray are all picture, we should define the Mat to store them.
    // of course, you should define the object to store the picture, it is a array in fact.
    // the imgGray is a outgoing param.
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    // the second, we can cast the picture to a type of GaussianBlur. and you can set the fuzzy
    // degree by adjusting the param. the third param define the fuzzy degree of picture. the big number
    // will set the more fuzzy degree.
    Mat imgBlur;
    GaussianBlur(img, imgBlur, Size(7, 7), 3, 0);

    // the third, we will use the edge detection. we should use the picture that be GaussianBlured as the 
    // src Mat param. it means we should cast the picture from imgBlur to imgCanny. the third and fourth param
    // can adjust the precision of the edge. the smaller number can get more precison edge.
    Mat imgCanny;
    Canny(imgBlur, imgCanny, 30, 100);

    // the fourth, we will expand the edge based on the imgCanny. the effection is bold the edge.
    // the kernel define the expand size, the bigger size of kernel, the more bold edge you will get.
    // and you should use the odd number as the size.
    Mat imgDil, imgErode;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(imgCanny, imgDil, kernel);

    // erosion the picture based on the imgDIL
    erode(imgDil, imgErode, kernel);
    //the bottom is to show the Mat array as a picture.
    imshow(" image BGR", img);
    imshow(" image GRAY", imgGray);
    imshow(" image Blur", imgBlur);
    imshow(" image CANNY", imgCanny);
    imshow(" image DIL", imgDil);
    imshow(" image ERODE", imgErode);
    */
    
    /** the next chapter we will learn how to resize the picture.
    string path = "resources/image.jpg";
    Mat img = imread(path);
    Mat imgResize1, imgResize2;

    // echo the size of picture img. the img object is a array in fact.
    cout << "the original size is : " << img.size() << endl;

    //resize
    resize(img, imgResize1, Size(650, 400));
    cout << "the size after resizing is : " << imgResize1.size() << endl;

    // you can alse resize the picture do not change the original aspect ratio.
    // the next code define the 50% width and height of original image.
    resize(img, imgResize2, Size(), 0.5, 0.5);
    cout << "the size after resizing is : " << imgResize2.size() << endl;

    // then we can define the custom picture size from the original image, it means we can screenshots
    // the size we want from the original, just like we want to get the face from a picture generally.
    // the next code means start from the left 270px to step 170px and from the top 110px to step 170px.
    Rect screenshots(270, 110, 170, 170);
    Mat imgCrop = imgResize2(screenshots);
    cout << "the size of screenshots is : " << screenshots.size() << endl;

    imshow(" image resize1", imgResize1);
    imshow(" image resize2", imgResize2);
    imshow(" image Crop", imgCrop);
    */
    
    /** the next we will create the picture by ourself.
    // we can define the picture using Mat class. the first and second param is the size of the picture
    // that you want to define. the third param is picture type. 8 mean 2^8=256, u mean unsinged, means from
    // 0 to 255, if signed, it will be from -128 to 127. the 3 means the picture we create is involved with three
    // matrix what is the gray picture. scalar define the BGR, 255,0,0 mean blue. you can define the number to get different color.
    // 255, 0, 255 is purple. blue and red.
    // you should keep in mind that
    // 255, 255, 255 is white
    // 0, 0, 0 is blank
    // 255, 0, 255 is purple.
    Mat img(512, 512, CV_8UC3, Scalar(255, 255, 255));

    //then we can graph a circle in the picture img
    // the img is the picture we will graph on it, and the seconde param is the center of the circle position.
    // the third param is the redius of the circle, the fourth param is the color of circle, the fifth param is
    // the bord of the circle. if you want to filled the circle, you should set the fifth param be FILLED.
    // the next code is graph a circle that center is the middle of the picture, and radius is 155, filled with blank color.
    circle(img, Point(256, 256), 155, Scalar(0, 0, 0), FILLED);

    // we can also graph a rectangle on the picture. you should graph the rectangle by defining the lefttop and rightbottom coordinates
    // the next code is graph a rectangle that lefttop coordinates is 101,101, the rightbottom coordinates is 411,411,
    // the color is purple and bord is 2px.
    rectangle(img, Point(101, 101), Point(411, 411), Scalar(255, 0, 255), 2);
    
    // the next we can graph a line on the picture.
    line(img, Point(101, 101), Point(411, 411), Scalar(0, 0, 255), 2);

    // we can alse graph the text on the picture.
    // you just need to define a start point. and the fourth param is the font, the fifth param is font size.
    // notice, the last param is bold of the text.
    putText(img, "WHOAMI", Point(185, 80), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));
    imshow("the picture we create", img);
    */
    
    /** the next we will learn how to distorted the picture.
    // what we want to do is normally show a distorted picture. we just need to give the coordinates.
    // we can use card to do it. we just need to give the coordinates in the picture.
    // you should give four point coordinates to graph a picture.
    string path = "resources/card.webp";
    Mat img = imread(path);
    //oh it is too big, so we resize it first.
    Mat imgResize;
    resize(img, imgResize, Size(), 0.3, 0.3);

    float w = 250, h = 350;
    Mat matrix, imgWrap;
    // first, we can define a array to store the four coordinates. we need to store it 
    // base on float. so we can use the Point2f that opencv give.
    // this point we can use the algorithm to recognize, or we can use the paint to get it.
    // then we must define the corrdinates dest we want to graph in a picture.
    // and the point coordinates should correspond to the src for picture.
    // the order is lefttop righttop leftbottom rightbottom. the lefttop is 0,0.
    // the param w and h we can define it by ourself, but we need to control the proportion
    Point2f src[4] = {{946, 572}, {1490, 669}, {821, 1443}, {1400, 1535}};
    Point2f dest[4] = {{0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h}};

    //we need two Mat type array to store the matrix and imgWrap.
    // notice, we should transform the picture size from src to dest, and return
    // the matrix about the data of dest size. then we can warp the picture from img 
    // to imgWrap base on the data array matrix, and we should give the point about w,h.
    // this just a simple rectangular image wrap, the important is about the wrap of a irregular image
    // if we get the coordinates, we can do something use it, just like we can graph something on it.
    matrix = getPerspectiveTransform(src, dest);
    warpPerspective(img, imgWrap, matrix, Point(w, h));

    // save the Mat array to diskdriver.
    imwrite("resources/imgresize.jpg", imgResize);

    long long unsigned int i;
    // we can graph the circle on the src array coordinates.
    for (i = 0; i < sizeof(src) / sizeof(Point2f); i++)
    {
        circle(img, src[i], 15, Scalar(0, 0, 255), FILLED);
    }
    imshow("the beauty card img", img);
    imshow("the beauty card imgResize", imgResize);
    imshow("the beauty card imgWrap", imgWrap);
    */
    
    /** the next chapter we will learn how to detect the color from picture. 
    // we should transform the picture from the basic BGR model to HSV model, in order to
    // we can detect the color from the picture easilier. we can use cvtColor. just like we 
    // used it in chapter1 to tranform from BGR to gray.
    // cvtColor(img, imgGray, COLOR_BGR2GRAY)
    Mat img, imgHSV, imgMask;
    string path = "resources/image8.jpg";
    img = imread(path);
    cvtColor(img, imgHSV, COLOR_BGR2HSV);

    // we have got the HSV picture, it is a picture model in fact. we can 
    // filter the color what we do not want and left the color what we want.
    // just like the cutout use chnnel in photoshop. but we must define the value based on
    // the HSV model, it is a rule in fact, we can define the min and max, then we can
    // get the middle of them about the picture. so the important is how to define the HSV value.
    int hmin = 0, smin = 110, vmin = 153;
    int hmax = 19, smax = 240, vmax = 255;
    // of course, we can define the value based on our experience, but opencv give us a more useful method.
    // that is the trackbar, we can adjust it in a real-time.

    // the next we will create the real-time adjustment trackbar for mask, 
    // filter the picture bases on the color for one HSV model picture.
    // the concept is simple, we should add a while loop to show the HSV picture, 
    // and you should define the inrange function in while function, and create the 
    // real-time adjustment window what is the createTrackbar out of the while function.
    // define a windows size is 640*200 and bind the name TrackBar
    // the concept is simple, we show the HSV picture forever, and use the pointer to
    // change the lower or upper value in time. so we can realize the real-time adjustment based on it.
    namedWindow("TrackBars", 1);
    // the fourth param define the max value we can adjust.
    // the third param is the value pointer to the &hmin. it is the current value show in the trackbar.
    // and it will change when we move the trackbar. 
    // and notice, if you want to realize the real-time adjustment, you need not to pass the zero to waitKey
    // beacause waitKey(0) mean fixed, the adjustment you do in trackbar will update in pointer but will not
    // reimread, beacause the process wait at there forever, unless you close all image window into a new while, 
    // or you will not change the filter rule. so you should give the waitKey a number greater than zero. just like 1.
    // once you find the most appropriate value rule, you should save the picture based on it.
    // similaly, we can detect the color what we want based on this method.
    createTrackbar("Hue Min", "TrackBars", &hmin, 179);
    createTrackbar("Hue Max", "TrackBars", &hmax, 179);
    createTrackbar("Sat Min", "TrackBars", &smin, 255);
    createTrackbar("Sat Max", "TrackBars", &smax, 255);
    createTrackbar("Val Min", "TrackBars", &vmin, 255);
    createTrackbar("Val Max", "TrackBars", &vmax, 255);

    while (true)
    {
        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);
        // we can use the lower and upper we defined, use mask model to filter the HSV picture.
        inRange(imgHSV, lower, upper, imgMask);
        imshow("image original", img);
        imshow("image hsv", imgHSV);
        imshow("image mask", imgMask);
        // we can alse understand the waitKey based on this, it is refresh rate, if the param is zero, 
        // the refresh rate is zero. so it is fixed. will not refresh. for movie, waitKey will show the 
        // the speed of playing picture, the greater number the smaller sppeed movie playing. similarly, 
        // if the params is zero, it will not refresh.
        waitKey(1);
    }
    */
    
    /** then we will learn how to detect the things base on the shape.
    // we will use the edge detection here. the step is read image, GaussianBlur it, Canny it.
    string path = "resources/image2.jpg";
    Mat img, imgGray, imgBlur, imgCanny, imgDil;
    img = imread(path);
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    // the size show the fuzzy degree of the GaussianBlur, the bigger number the more fuzzy degree.
    // if you want to show more information about the edge of the picture, you should use the smaller number for size
    // the fourth param is the 
    GaussianBlur(imgGray, imgBlur, Size(7, 7), 3, 0);

    // detect the edge based on the imgGray
    Canny(imgBlur, imgCanny, 25, 75);

    // expand the canny based on the imgCanny, it is border the edge in fact
    // the second param is the boder value for the edge. why we need to expand the canny?
    // beacause it can handle the clearance in canny.
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgCanny, imgDil, kernel);

    // util here, we have done the reprocessing for the picture, then we should define the fucntion
    // to detect the contours, what is the shape.

    // we can get all contours in the picture used the function getContour we defined, it used the findContour
    // and drawContours fucntion and so on in fact. the function find all the contours based on the imgDil
    // using findContour fucntion first, then draw the contours you want to show in your original picture 
    // using drawContour function.
    // these contours data we used the vector to store it.
    getContours(imgDil, img);

    imshow("image original", img);
    */
       
    // imshow("image gray", imgGray);
    // imshow("image blur", imgBlur);
    // imshow("image canny", imgCanny);
    // imshow("image dil", imgDil); 

    /* this chapter we will learn how to detetct the face.
    // thie function we will realize it based on the opencv2/objdetect.hpp
    // we need to use the headfile opencv2/objdetect.hpp and xml file haarcascade_frontalface_default.xml
    // the seconde file we should give the path in our diskdrive. this file is loaded when you 
    // install the opencv2, the path of pip opencv-python is in C:\users\80521\appdata\local\programs\python\python38\Lib\site-packages\cv2\data
    // the path of opencv using cmake is in D:\development_app2\opencv\source\opencv\data
    Mat img;
    string path = "resources/image8.jpg";
    img = imread(path);

    // define a loader to load the faceCascade xml file.
    CascadeClassifier faceCascade;

    // this xml file will load the net resource, then give the face feature to 
    // help us to find the rectangular coordinates.
    faceCascade.load("resources/haarcascade_frontalface_default.xml");
    // fault tolerance
    if (faceCascade.empty()) { cout << "xml file not loaded" << endl; };
    // define a vector about Rect to store the faces data pointer.
    vector<Rect> faces;
    faceCascade.detectMultiScale(img, faces, 1.1, 10);
    cout << faces.size() << endl;
    // we should graph the boundary for each face have detected.
    // the Rect is the rectangular data type. the faces is a rectangular vector
    // the faces rectangular data saves in the vector.
    for (int i = 0; i < faces.size(); i++)
    {
        // read one face data, and graph the boundary based on the rectangular
        rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0), 3);
    }

    imshow("image original", img); 
    waitKey(0);
    */

    /*// then we can open the camera and detect the face forever.
    Mat img;
    VideoCapture capture(0);
    CascadeClassifier faceCascade;
    faceCascade.load("resources/haarcascade_frontalface_default.xml");
    vector<Rect> faces;
    while (capture.read(img))
    {   

        faceCascade.detectMultiScale(img, faces, 1.1, 10);
        for (int i = 0; i < faces.size(); i++)
        {
            rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0), 3);
        }
        imshow("the camera", img);
        waitKey(1);    
    }
    */
    
    // the next chapter we will learn how to virtual graph.
    
    // detectPlates();


    // the next we will learn the opencv basis knowledge
    // you can read a image as the gray model, just like the next code.
    // the second param is model, the default is 1 GBR, you should pass 0 if you want to show gray.
    /*
    Mat color = imread("resources/image1.jpg");
    Mat gray = imread("resources/image1.jpg", 0);

    if (!color.data)
    {
        cout << "could not open or find the image" << endl;
        return -1;
    }
    // imshow("gray image", gray);
    // Vec3b is the class to store the GBR, we can use  the Mat::at<typename>(row, col)
    // to get a Vec3b what stored the GBR three element. 
    int myRow = color.rows - 1;
    int myCol = color.cols - 1;
    Vec3b object = color.at<Vec3b>(myRow, myCol);

    for (int i = 0; i < 3; i++)
        cout << (int)object[i] << endl;
    
    VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
        return -1;
    namedWindow("video", 1);
    for (;;)
    {
        Mat frame;
        if (cap.read(frame))
        {
            imshow("video", frame);
        }
        if (waitKey(30) >= 0)
        {
            break;
        }
    }

    // page 40
    // release the cap;
    cap.release();
    waitKey(0);
    */
   
    // windows you created, WINDOW_AUTOSIZE is default param. you can reset the param.
    // param 0 means you can resize the window. you can resize the window as follow
    /*     
    namedWindow("the beauty woman huliena", 0);
    resizeWindow("the beauty woman huliena", 900, 500);
    string path = "resources/美女胡列娜.png";
    Mat image = imread(path);
    cout << image.size() << endl;
    uchar z = get_scale_value(image, 0, 0);
    cout << (int)z << endl;
    cout << (int)image.ptr<uchar>(0)[0] << endl;
    imshow("the beauty woman huliena", image); 
    */

    string path = "resources/美女胡列娜.png";
    Mat image, imageGray, resizeImage;
    image = imread(path);
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    cout << "this resolution of the gray image is " << imageGray.size << endl;
    Mat output_image = scale(imageGray, 360, 600);
    // of course, you can also use the method resize that opencv has provided.
    resize(image, resizeImage, Size(), 0.5, 0.5);
    Mat output_image_binary = binary_linear_scale(output_image, 720, 1200);
    Mat output_image_linear = binary_linear_scale(output_image, 720, 1200);
    imshow("the gray image", imageGray);
    imshow("the gray image after scaling", output_image);
    imshow("the gray image after resizing", resizeImage);
    imshow("the gray image after bianryScaling", output_image_binary);
    imshow("the gray image after linearScaling", output_image_linear);

    waitKey(0);

    // notice, you should destroy all the windows you have created at end.
    destroyAllWindows();
    system("pause");
    return 0;
}
