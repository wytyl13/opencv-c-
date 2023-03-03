#include "../include/general.h"
#include "../include/linearInterpolation.h"
#include "../include/noise.h"
#include "../include/transformUsedAffineMatrix.h"
#include "../include/someSuperApplication.h"
#include "../include/eigen.h"
#include "../include/grayLevelTransform.h"
#include "../include/bitOperation.h"
#include "../include/spatialFilter.h"
#include "../include/faceApplication.h"
#include "../include/featureInImage.h"
#include "../include/wordDetect.h"
#include "../include/base64.h"
#include "../include/opticalCharacterRecognition.h"
#include <time.h>

// typedef enum ENUM
// {
//     A,
//     B,
//     C
// } ENUMCLASS;

int main(int argc, char const *argv[])
{
    srand((unsigned)time(NULL)); // define the randSeed based on the current time.
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

    // string path = "resources/美女胡列娜.png";
    // Mat image, imageGray, resizeImage;
    // image = imread(path);
    // cvtColor(image, imageGray, COLOR_BGR2GRAY);
    // cout << "this resolution of the gray image is " << imageGray.size << endl;
    // Mat output_image = scale(imageGray, 360, 600);
    // // of course, you can also use the method resize that opencv has provided.
    // resize(image, resizeImage, Size(), 0.5, 0.5);
    // Mat output_image_binary = binary_linear_scale(output_image, 720, 1200);
    // Mat output_image_linear = binary_linear_scale(output_image, 720, 1200);
    // imshow("the gray image", imageGray);
    // imshow("the gray image after scaling", output_image);
    // imshow("the gray image after resizing", resizeImage);
    // imshow("the gray image after bianryScaling", output_image_binary);
    // imshow("the gray image after linearScaling", output_image_linear);


    // --------------------test image arithmetic-------------------------------

    // string path = "resources/美女胡列娜.png";
    // Mat image, imageGray, resizeImage;
    // image = imread(path);
    // cvtColor(image, imageGray, COLOR_BGR2GRAY);
    // resizeImage = binary_linear_scale(imageGray, 720, 1200);
    // Mat dstImage = resizeImage;
    // we have defined the function about gaussion and saltPerrer noise for on picture, and test successfully, then we should
    // test the image arithmetic, but we failed to add two Mat, because we have not found the method to breakThrough the element
    // range from 0 to 255. we have tested CV_64FC1 and so on, they always can not breakThrough the max range 255.
    // then, we will insight into the image arithmetic.
/*     Mat src = Mat::ones(dstImage.rows, dstImage.cols, CV_8UC1);
    Mat averageImage = Mat::zeros(dstImage.rows, dstImage.cols, CV_64FC3); // 32 bit float to store the result that all image element added.
    // Mat averageImage;
    add(dstImage, src, averageImage);
    add(dstImage, averageImage, averageImage);
    for (int i = 0; i < 10; i++)
    {
        cout << (int)dstImage.ptr<uchar>(5)[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << (int)averageImage.ptr<uchar>(5)[i] << " ";
    }
    cout << endl;
    for (int i = 0; i < 50; i+=5)
    {
        gaussionNoise(resizeImage, dstImage, 0, i);

        // add(dstImage, result, result);
        // add dstImage and averageImage.
        // averageImage += dstImage;
    
    }
    // averageImage /= 100;
    // averageImage.convertTo(averageImage, CV_8UC1);
    // imshow("the image after denoising", averageImage);
    // imshow("the image after denoising" + i, gaussionNoise);
    // averageImage.convertTo(averageImage, CV_8UC1); */
    

    // notice, if you used F5 to run this file in vscode. you should use the relative
    // path of the current project.
    // if you run this file in windows terminal, you should use the relitive path
    // of the current terminal path.
    // string path = "../resources/美女胡列娜.png";
    // Mat image, imageGray, resizeImage;
    // image = imread(path, CV_32FC1);
    // cvtColor(image, imageGray, COLOR_BGR2GRAY);
    // resizeImage = binary_linear_scale(imageGray, 360, 600);
    
    // Mat outputImage;
    // imshowMulti function and elementOperation function test
    /*     
    elementOperation(resizeImage, outputImage);
    vector<Mat> imageVector(2);
    imageVector[0] = resizeImage;
    imageVector[1] = outputImage;
    imageVector.push_back(outputImage);
    imageVector.push_back(resizeImage);
    imageVector.push_back(outputImage);
    imageVector.push_back(resizeImage);
    imageVector.push_back(outputImage);
    imageVector.push_back(resizeImage);
    imageVector.push_back(outputImage);
    string str = "the compared original gray image and element operations information";
    // the second param in nameEindow is the right that resize the original image.
    // notice, the default param is not 0, it can not resize the original image size
    // based on changed the window size, so you can change the original image size that the window have
    // displayed. 
    namedWindow(str, 0);
    imshowMulti(str, imageVector); 
    */
    
    // test the affine transformation function that opencv provided.
    // we will test the offical function. we have defined the function that used official affine transformation.
    // affineTransformUsedOfficial(imageGray, outputImage);
    // this function is generally used for scale and rotation by changing the third param and fourth param.
    // the third param is rotation angle, the fourth param is the scale rate.
    // transformUsedOfficialBasedOnSpecificParam(imageGray, outputImage, 45, 0.2);
    // resize(outputImage, outputImage, Size(600, 360));
    // string str = "the difference between original and affine transformation";
    // namedWindow(str, 1);
    // rotationUsedAffineMatrix(resizeImage, outputImage, 90, 1.0);
    // this param will return the original, because the size is equal to the original image size.
    // transformUsedOfficialBasedOnThreePoint(resizeImage, outputImage, Size2f(0, 0), Size2f(0, 1), Size2f(1, 0));
    // transformUsedOfficialBasedOnThreePoint(resizeImage, outputImage, Size2f(0.1, 0.1), Size2f(0.2, 0.7), Size2f(0.7, 0.1));
    // imshow("original image", resizeImage);
    // imshow(str, outputImage);


    // vector<Mat> vectorImages;
    // vectorImages.push_back(imageGray);
    // vectorImages.push_back(outputImage);

    // imageRotationSimple(resizeImage, 0);
    // imshow("test", resizeImage);
    // imwrite("D:/development_code_2022-9-10/vscode/opencv/resources/test.tiff", outputImage);


    
    // the custom affine transformation function test
    // identityTransformMatrix(resizeImage, outputImage);
    // string str = "show all images";
    // namedWindow(str, 0);
    // vector<Mat> vectorImages;
    // vectorImages.push_back(resizeImage);
    // vectorImages.push_back(outputImage);
    // imshowMulti(str, vectorImages); 

    // --------------------test image arithmetic-------------------------------

    // --------------------test Eigen-------------------------------
    /*
    defineSpecificMatrix();
    Mat image, imageGray;
    image = imread("../resources/image4.png");
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    imshow("beauty", imageGray);
    // MatrixXi matrix(imageGray.rows, imageGray.cols);
    MatrixXi matrix;
    // you can use cv2eigen or eigen2cv to transform between these two array object.
    cv2eigen(imageGray, matrix);
    // notice, the difference between the rows and cols function of matrix and Mat
    // the former must use (), because it is the function of Matrix.
    // the last can not use (), because it is the attribute of Mat.
    cout << matrix.rows() << "," << matrix.cols() << endl;
    cout << imageGray.rows << "," << imageGray.cols << endl;
    */

    // transform from Matrix for eigen to Mat for opencv
    // define a 4*4 random int data matrix
    /* 
    Matrix4i matrix = Matrix4i::Random();
    Mat image;
    eigen2cv(matrix, image);
    cout << matrix << endl;
    cout << image << endl; 
    */
    // we will test matrix multi used Matrix class in Eigen
    // because we have tested the Mat operate used Mat, but it can not work.
    // because the matrix has not many limit condition like Mat, so we can calculate
    // the matrix operation used matrix, then transform the result to Mat to show this image.
    // just like calculate the good image used the images with noise, you can calculate the
    // mean of all the noise image, but the Mat object has the limit about the size of data type.
    // so we can calculate the mean first, then transform from matrix to Mat.
    /* Mat image, imageGray, resizeImage, noiseImage1, noiseImage2, noiseImage3, noiseImage4, noiseImage5;
    image = imread("../resources/hln.png");
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    resize(imageGray, resizeImage, Size(600, 360));

    gaussionNoise(resizeImage, noiseImage1, 10, 50);
    gaussionNoise(resizeImage, noiseImage2, 10, 55);
    gaussionNoise(resizeImage, noiseImage3, 10, 60);
    gaussionNoise(resizeImage, noiseImage4, 10, 65);
    gaussionNoise(resizeImage, noiseImage5, 10, 70);

    MatrixXd matrix1, matrix2, matrix3, matrix4, matrix5;
    cv2eigen(noiseImage1, matrix1);
    cv2eigen(noiseImage2, matrix2);
    cv2eigen(noiseImage3, matrix3);
    cv2eigen(noiseImage4, matrix4);
    cv2eigen(noiseImage5, matrix5);

    MatrixXd result = (matrix1 + matrix2 + matrix3 + matrix4 + matrix5) / 5;
    printf("row = %d, col = %d\n", result.rows(), result.cols());
    // if you want to show one gray image, you should use the 8UC1 data type to store each element.
    // it means we should transform the each element of matrix from double to int.
    // then we will transform from MatrixXd to MatrixXi
    MatrixXi result_int = result.cast<int>();
    Mat result_image;
    eigen2cv(result_int, result_image);
    result_image.convertTo(result_image, CV_8UC1);
    namedWindow("test image", 1);
    imshow("test image", resizeImage);
    imshow("test image", noiseImage5);
    imshow("result image", result_image); */
    // but we have failed to reduction the noiseImage. because we have defined a large var.
    // but we have finished this application of the method.
    // then, we will learn the rest of the digital image processing.
    // --------------------test Eigen-------------------------------
    
    // --------------------test grayLevelTransform-------------------------------
/*     string str = "compare the original and result image";
    namedWindow(str, 1);
    Mat image, imageGray, resizeImage;
    image = imread("../resources/hln.png");
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    resize(imageGray, resizeImage, Size(600, 360));  
    Mat outputImage;
    reverseTransform(resizeImage, outputImage);
    vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    vectorImages.push_back(outputImage);
    imshowMulti(str, vectorImages); */
    // --------------------test grayLevelTransform-------------------------------
    // --------------------test logarithmic transform and linear scaling-------------------------------
    // string str = "compare the original and result image";
    // namedWindow(str, 1);
    // Mat image, imageGray, resizeImage;
/*     image = imread("../resources/fourier.png");
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    resize(imageGray, resizeImage, Size(600, 360));
    Mat outputImage;
    logarithmicAndLinearScaling(resizeImage, outputImage);
    vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    vectorImages.push_back(outputImage);
    imshowMulti(str, vectorImages); */


/*     image = imread("../resources/hln.png");
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    resize(imageGray, resizeImage, Size(600, 360));
    Mat outputImage;
    linearScaling(resizeImage, outputImage);
    vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    vectorImages.push_back(outputImage);
    imshowMulti(str, vectorImages); */


    // image = imread("../resources/hln.png");
    // cvtColor(image, imageGray, COLOR_BGR2GRAY);
    // resize(imageGray, resizeImage, Size(600, 360));
    // Mat outputImage;
    // linearScalingBaseTwoPoint(resizeImage, outputImage);
    // vector<Mat> vectorImages;
    // vectorImages.push_back(resizeImage);
    // vectorImages.push_back(outputImage);
    // imshowMulti(str, vectorImages);
/*     gamaTransformAndLinearScaling(resizeImage, outputImage, 30, 0.6);
    vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    vectorImages.push_back(outputImage);
    gamaTransformAndLinearScaling(resizeImage, outputImage, 30, 0.5);
    vectorImages.push_back(outputImage);
    gamaTransformAndLinearScaling(resizeImage, outputImage, 30, 0.4);
    vectorImages.push_back(outputImage);
    imshowMulti(str, vectorImages); */
    // --------------------test logarithmic transform and linear scaling-------------------------------

    // --------------------test bit operation--------------------------
    // bitWise(resizeImage, outputImage);
    // imshow(str, outputImage);
    // --------------------test bit operation--------------------------

    // --------------------test draw and screenShots and gray level layered--------------------------
/*     Point start(0, 0);
    Point end(100, 100);
    // drawLines(resizeImage, start, end);
    Point one(0, 0);
    Point two(100, 0);
    Point three(100, 100);
    Point four(0, 100);
    Point five(50, 150);
    vector<Point> vectorPoints;
    vectorPoints.push_back(one);
    vectorPoints.push_back(two);
    vectorPoints.push_back(three);
    vectorPoints.push_back(five);
    vectorPoints.push_back(four);
    drawPolygon(resizeImage, vectorPoints);
    imshow(str, resizeImage); */
 /*    Point one(0, 0);
    Point two(100, 0);
    Point three(100, 100);
    Point four(0, 100);
    Point five(50, 150);
    vector<Point> vectorPoints;
    vectorPoints.push_back(one);
    vectorPoints.push_back(two);
    vectorPoints.push_back(three);
    vectorPoints.push_back(five);
    vectorPoints.push_back(four);
    // grayLayeredBasedPoints(resizeImage, outputImage, vectorPoints, 0);
    Rect rect = Rect(one, three);
    cout << rect.size() << endl;
    // screenShots(resizeImage, outputImage, rect);

    // cutImage(resizeImage, outputImage, vectorPoints);
    grayLayeredBasedPoints(resizeImage, outputImage, vectorPoints, 1);
    imshow(str, outputImage); */

    // grayLayeredBasedValue(resizeImage, outputImage);
    // grayLayeredBasedBitPlane(resizeImage, outputImage, 1);
    // vector<Mat> vectorImages;
    // vectorImages.push_back(resizeImage);
    // vectorImages.push_back(outputImage);
    // imshowMulti(str, vectorImages);

/*     vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    // get 8 bit plane.
    map<int, Mat, compareMap> mapBitPlanes; */

    // just like this case, we have got the best results used 7 bit plane and 8 bit plane
    // to reconstruct the original image.
    // if you add the lower bit plane, just like 6 and 5 bit plane, the effect will be worse.
/*     for (int i = 7; i < 9; i++)
    {
        grayLayeredBasedBitPlane(resizeImage, outputImage, i);
        mapBitPlanes.insert(pair<int, Mat>(i, outputImage));
        vectorImages.push_back(outputImage);
    }
    Mat resultMat;
    reconstructImageBasedBitPlane(resultMat, mapBitPlanes);
    vectorImages.push_back(resultMat);
    imshowMulti(str, vectorImages); */
    // --------------------test draw and screenShots and gray level layered--------------------------
    // --------------------test histogram transform-----------------------------
    // imshow(str, resizeImage);
/*     double *listOriginal = getDistribution(resizeImage);
    // printArray(list);
    Mat histogramMatOriginal, histogramMatNew;
    vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    getHistogramMat(listOriginal, histogramMatOriginal);
    vectorImages.push_back(histogramMatOriginal);
    histogramEqualizeTransformation(resizeImage, outputImage);
    vectorImages.push_back(outputImage);
    double *listNew = getDistribution(outputImage);
    getHistogramMat(listNew, histogramMatNew);
    vectorImages.push_back(histogramMatNew);
    imshowMulti(str, vectorImages); */
    // we have successful tested the histogramTransform, it can enhance contrast.
    // and the efficient used it is very good.
    // you can find that the histogram transform has a lot to do with the the conrast of one image.
/*  // of course, you can also use the official function equalizeHist that opencv has defined.
    equalizeHist(resizeImage, outputImage);
    imshow(str, outputImage); */
    

    // we will test the histogram matching function.
    // you can find it is euqal to the original image if you used the pure blank image as the objectImage.
    // Mat obj_image = imread("../resources/darkerImage.webp");
    // Mat gray_image, objectImage, equalizationMat;
    // Mat equalizationMat_equalization_original, equalizationMat_matching_original;
    // Mat equalizationMat_original, equalizationMat_object;
    // Mat matching_original, equalization_original, matching_equalization_original;
    // Mat matching_object, equalization_object, matching_equalization_object;
    // cvtColor(obj_image, gray_image, COLOR_BGRA2GRAY);
    // resize(gray_image, objectImage, Size(720, 360));
    // objectImage = Mat::zeros(Size(100, 100), CV_8UC1);
    // objectImage = Mat::ones(Size(100, 100), CV_8UC1);
    // Point leftUpper = Point(0, 0);
    // Point rightUpper = Point(100, 0);
    // Point rightDown = Point(100, 100);
    // Point leftDown = Point(0, 100);
    // vector<Point> vectorPoints;
    // vectorPoints.push_back(leftUpper);
    // vectorPoints.push_back(rightUpper);
    // vectorPoints.push_back(rightDown);
    // vectorPoints.push_back(leftDown);
    // fillPoly(objectImage, vectorPoints, Scalar::all(255));

    /* str = "compare the histogram and image of the original and other transformation";
    vector<Mat> vectorImages_compare1;
    vectorImages_compare1.push_back(resizeImage);
    vectorImages_compare1.push_back(objectImage);
    getHistogramMatBasedOnInputImage(resizeImage, equalizationMat_original);
    vectorImages_compare1.push_back(equalizationMat_original);
    getHistogramMatBasedOnInputImage(objectImage, equalizationMat_object);
    vectorImages_compare1.push_back(equalizationMat_object);
    histogramEqualizeTransformation(resizeImage, equalization_original);
    vectorImages_compare1.push_back(equalization_original);
    getHistogramMatBasedOnInputImage(equalization_original, equalizationMat_equalization_original);
    vectorImages_compare1.push_back(equalizationMat_equalization_original);
    histogramMatchingTransformation(resizeImage, objectImage, matching_original);
    vectorImages_compare1.push_back(matching_original);
    getHistogramMatBasedOnInputImage(matching_original, equalizationMat_matching_original);
    vectorImages_compare1.push_back(equalizationMat_matching_original);
    imshowMulti(str, vectorImages_compare1);

    str = "compare the image of resizeImage, any transformation involved matching_original, equalization_original, matching_equalization";
    histogramMatchingTransformation(equalization_original, objectImage, matching_equalization_original);
    vector<Mat> vectorImages_compare2;
    vectorImages_compare2.push_back(resizeImage);
    vectorImages_compare2.push_back(matching_original);
    vectorImages_compare2.push_back(equalization_original);
    vectorImages_compare2.push_back(matching_equalization_original);
    imshowMulti(str, vectorImages_compare2);

    str = "compare the image of objectImage, any transformation involved matching_original, equalization_original, matching_equalization";
    histogramMatchingTransformation(objectImage, resizeImage, matching_object);
    histogramEqualizeTransformation(objectImage, equalization_object);
    histogramMatchingTransformation(equalization_object, resizeImage, matching_equalization_object);
    vector<Mat> vectorImages_compare3;
    vectorImages_compare3.push_back(objectImage);
    vectorImages_compare3.push_back(matching_object);
    vectorImages_compare3.push_back(equalization_object);
    vectorImages_compare3.push_back(matching_equalization_object);
    imshowMulti(str, vectorImages_compare3); */

    // histogramTransformationLocal(resizeImage, outputImage, 4);
    // imshow("test the local histogram transformation", outputImage);
    // double *distribution = getEqualizationDistribution(resizeImage);
    // printOneArrayPointer(distribution);
/*     str = "compare the original image and local histogram equalization image";
    histogramTransformationLocal(resizeImage, outputImage, 6);
    vector<Mat> vectorImages;
    vectorImages.push_back(resizeImage);
    vectorImages.push_back(outputImage);
    imshowMulti(str, vectorImages); */

/*     thread first_thread(thread_function, 1);
    thread second_thread(thread_function, 2);

    first_thread.join();
    second_thread.join(); */
    // cpp is simpled to use the thread. create the thread object means the thread started
    // join or detach means the thread end. the join means the process will block at here
    // and util the thread end. detach will not wait the thread end. but
    // these two keyword will all free the thread. the thread will enjoy all the code in the main
    // after the thread creation.
    // you can find each thread is separate during runing, and they will rob the cpu resoueces.
    // if you call the cout function or other function, the other thread will rob the resources.
    // so you will get the print content like as follow. "子线程子线程2开始执行1开始执行"
    // the first thread rob the standard output resources and print "子线程", when the first thread
    // want to print the other content, the second thread has robbed the cpu resouces and print all
    // content it want to print. but when the second thread want to print the endl, the first thread
    // robbed the cpu resources and print the other content, at the end, these two thread all
    // print the endl; because we has free the thread used join keyword, so the main thread will not
    // rob the resource before all sub thread run end. the main thread will rob the cpu resources
    // if you used the detach keyword to free the sub thread.
    // cout << "main thread" << endl;

    // you can find the function used thread will waste less time, the thread function waste 4s
    // if the kernel is 16, but the general function will wast 17s. the difference is 13s.
    // if you adjust the parameter about the thread numbers, you will waste less time.
    // we have test the 12 thread, it just wasted 1s. the efficient is equal to the general local histogram function.
/*     time_t start_thread = time(NULL);
    Mat outputImage_thread, outputImage_general;
    histogramTransformationLocalThread(objectImage, outputImage_thread, 8, 4);
    time_t end_thread = time(NULL);
    time_t time_thread = end_thread - start_thread;
    cout << "the local histogram transformation used thread has wasted " << time_thread << "s" << endl;
    time_t start_general = time(NULL);
    histogramTransformationLocal(objectImage, outputImage_general, 8);
    time_t end_general = time(NULL);
    time_t time_general = end_general - start_general;
    cout << "the local histogram transformation without thread has wasted " << time_general << "s" << endl;
    cout << Mat(objectImage, Range(0, 1), Range(350, 370)) << endl;
    // cout << Mat(outputImage, Range(0, 1), Range(350, 370)) << endl;
    vector<Mat> vectorImages;
    vectorImages.push_back(objectImage);
    vectorImages.push_back(outputImage_thread);
    vectorImages.push_back(outputImage_general);
    imshowMulti(str, vectorImages); */
    // we have tested the histogram equalization local function used multi threads, it also has some problem expect to slove.
    // then, we will test the other content of the digital image processing.
    // of course, you can also test the histogram matching local function. we are here to omit it.

    // quickly initialize one Mat
    // mean 10*0.25+20*0.5+30*0.25 = 2.5+10+7.5=20
    // variance (10-20)^2*0.25+(20-20)^2*0.5+(20-20)^2*0.5+(30-20)^2*0.25 = 25+25=50
/*     uchar m[2][2] = {{10, 20}, {20, 30}};
    Mat testMat = Mat(2, 2, CV_8UC1, m);
    getMeanAndVarianceBaseOnMat(testMat, array);
    cout << array[0] << " " << array[1] << endl;
    double *distribution = getDistribution(testMat);
    printOneArrayPointer(distribution); */
/*     double k[4] = {0.0, 0.99999, 0.0, 0.0000009};
    // LocalWithStatistics(objectImage, outputImage, 2, k);
    LocalWithStatistics(objectImage, outputImage, 4, k);
    vector<Mat> vectorImages;
    vectorImages.push_back(objectImage);
    vectorImages.push_back(outputImage);
    imshowMulti(str, vectorImages); */

    #if ISOPENHISTOGRAMTRANSFORM
    Mat darkImage = imread("../../resources/darkerImage.webp");
    Mat darkImage_ = imread("../../resources/darkerImage.jpeg", 0);
    Mat grayImage, equalizeImage, localStatisticImage, matchingImage;
    cvtColor(darkImage, grayImage, COLOR_BGR2GRAY);
    histogramEqualizeTransformation(grayImage, equalizeImage);
    histogramTransformationLocal(grayImage, outputImage, 3);
    double k[4] = {0.0, 0.8, 0.0, 0.8};
    LocalWithStatistics(grayImage, localStatisticImage, 7, k);
    histogramMatchingTransformation(equalizeImage, darkImage_, matchingImage);
    vector<Mat> vectorImages;
    vectorImages.push_back(grayImage);
    vectorImages.push_back(equalizeImage);
    vectorImages.push_back(outputImage);
    vectorImages.push_back(localStatisticImage);
    vectorImages.push_back(matchingImage);
    string str = "compare histogram transformation";
    namedWindow(str, 1);
    imshowMulti(str, vectorImages);
    #endif
    // --------------------test histogram transform-----------------------------

    // --------------------test spatial filter device -----------------------------
    // Mat inputImage = imread("../resources/hln.png", 0);
    // Mat outputImage;
    // vector<Mat> vectorImages;
    // officialFilterTest(inputImage, outputImage, SFD::FUZZY);
    // vectorImages.push_back(inputImage);
    // vectorImages.push_back(outputImage);
    // string str = "resharpen the image";
    // // imshowMulti(str, vectorImages);
    // imshow("original image", inputImage);
    // imshow("sharpen image", outputImage);
/*     Mat inputImage2 = imread("../resources/dusk.jpg", 1);
    officialImageMixTest(inputImage, inputImage2, outputImage, 0.5);
    imshow("mix image", outputImage); */
    // --------------------test spatial filter device -----------------------------




    // --------------------test super application-----------------------------
    string windowName = "face detect movie";
    string path = "../../resources/xiyou.mp4";
    // faceDetectMovie(windowName, path, face_cascade, eye_cascade);

    Mat faceImage = imread("../../resources/faces.jfif");
    Mat outputImage;
    CascadeClassifier face_cascade;
    CascadeClassifier eye_cascade;
    // faceDetectImage(faceImage, outputImage, face_cascade, eye_cascade);
    // imshow("face detect image test", outputImage);

/*     Mat faceImage = imread("../resources/faces.jpeg");
    Mat inputImage;
    resize(faceImage, inputImage, Size(720, 360));
    faceDetectImage(inputImage, outputImage, face_cascade);
    imshow(str, outputImage); */


    // CascadeClassifier face_cascade;
    // CascadeClassifier eye_cascade;
    // string windowName = "movie play device";
    // faceDetectMovie(windowName, "../resources/yuanyuan.mp4", face_cascade, eye_cascade);

    // Mat image1 = imread("../resources/hln.webp", 0);
    // Mat image2 = imread("../resources/yy2.webp", 0);
    // faceRecognition(image1, image2);
    
    // printf("SIFT IS %d\n", FD::SIFT);
    // open a picture in gray mode.
    // Mat image = imread("../resources/hln.webp", 0);
    // imshow("original image", image);

    // if you scale the matrix, the feature vector is constant, the feature value is 
    // also scaled. 
/*     Mat eigen_value, eigen_vector;
    Mat data = (Mat_<double>(2, 2) << 1, 2, 2, 1);
    eigen(data, eigen_value, eigen_vector);
    cout << eigen_value << endl;
    cout << eigen_vector << endl; */
    // BINARY THE IMAGE
    // threshold(inputImage, outputImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // imshow("threshold image", outputImage);

    // test the boundary tranck and pca and calculate the center point based on 
    // the feature points set of one detected object in one image.
/*     Mat inputImage = imread("../resources/123.jfif");
    resize(inputImage, inputImage, Size(720, 360));
    Mat outputImage;
    boundaryTrackUsedOfficial(inputImage, outputImage); */


    Mat oceanImage = imread("../../resources/ocean.webp");
    Mat oceanGrayImage;
    Mat ballImage = imread("../../resources/ball.webp");
    Mat heyingImage = imread("../../resources/heying.jpg");
    Mat oceanFaceImage = imread("../../resources/oceanFace.webp");
    Mat houseImage = imread("../../resources/house.jpg");
    cvtColor(oceanImage, oceanGrayImage, COLOR_BGR2GRAY);
    resize(oceanFaceImage, oceanFaceImage, Size(500, 500));
    resize(houseImage, houseImage, Size(500, 500));
    vector<Mat> vectorImages;
/*     faceFeatureDetectUsedOfficial(oceanImage, outputImage);
    imshow("feature detect test one", outputImage);
    faceFeatureDetectUsedOfficial(oceanFaceImage, outputImage);
    imshow("feature detect test two", outputImage);
    faceFeatureDetectUsedOfficial(houseImage, outputImage);
    imshow("house feature detect test", outputImage); */
/*     featureDetectUsedOfficial(oceanFaceImage, outputImage);
    imshow("oceanface feature detected image", outputImage);
    boundaryDetectUsedOfficial(oceanFaceImage, outputImage);
    imshow("oceanface boundary detected image", outputImage);
    featureDetectUsedOfficial(houseImage, outputImage);
    imshow("house feature detected image", outputImage);
    boundaryDetectUsedOfficial(houseImage, outputImage);
    imshow("house boundary detected image", outputImage); */
    // Mat resizeImage;
    // faceDetectUsedDlib(heyingImage, outputImage, DLIB::MMOD);
    // resize(outputImage, resizeImage, Size(1400, 700));
    // imshow("dlib detected face image", resizeImage);

    // getFaceSamplesFromMovie("../resources/movie/gutianle.mp4", "../resources/trainSample/gtl");
    /* vector<string> vectorPath;
    int count = 0;
    getAllFileFromDirAndCreatTrainData("../../resources/trainSample", vectorPath, \
        "../../resources/labelFile/gyygtl.txt", count);
    faceRecognitionUsedEigenFace("../../resources/labelFile/gyygtl.txt", "../../resources/movie/predict.mp4"); */

    #if ISOPENFACEAPPLICATION
    // test face recognition about dlib.
    faceDetectUsedDlib(oceanImage, outputImage, DLIB::SHAPE68);
    imshow("ocean image", outputImage);
    const string dirPath = "../../resources/trainSample/resnet_src/src";
    const string targetImagePath = "../../resources/trainSample/resnet_src/target/target.jpg";
    faceImageRecognitionUsedDlib(dirPath, targetImagePath);
    const string dirPath = "../../resources/trainSample/resnet_src/src";
    const string targetMoviePath = "../../resources/trainSample/resnet_src/target/movie.mp4";
    faceMovieRecognitionUsedDlib(dirPath, targetMoviePath);
    #endif
    // --------------------test super application-----------------------------

    // --------------------return to test the spatial filter-----------------------------
    #if ISOPENSPATIALFILTER
    // officialFilterTest(oceanImage, outputImage, FUZZYKERNEL);
    // imshow("test", outputImage);
    string str = "compare the filter";
    Mat noiseImage = cv::imread("../../resources/noiseImage.webp");
    Mat noiseGrayImage;
    cvtColor(noiseImage, noiseGrayImage, COLOR_BGR2GRAY);
    #if 0
    Mat sharpenImage, fuzzyImage, image_, smoothImage, smoothImageGaussian;
    Mat fuzzySeparated, smoothSeparated;
    vector<Mat> vectorMats;
    spatialFilterOperation(oceanGrayImage, outputImage, SHARPENKERNEL);
    vectorMats.push_back(oceanGrayImage);
    vectorMats.push_back(outputImage);
    officialFilterTest(oceanGrayImage, sharpenImage, SHARPENKERNEL);
    vectorMats.push_back(sharpenImage);
    spatialConvolution(oceanGrayImage, fuzzyImage, SHARPENKERNEL_);
    vectorMats.push_back(fuzzyImage);
    spatialFilter(oceanGrayImage, image_, FUZZYKERNEL, CONVOLUTION);
    vectorMats.push_back(image_);
    spatialFilter(oceanGrayImage, smoothImage, SMOOTHKERNELCASSETTE, CONVOLUTION);
    vectorMats.push_back(smoothImage);
    spatialFilter(oceanGrayImage, smoothImageGaussian, SMMOTHKERNELGAUSSIAN, CONVOLUTION);
    vectorMats.push_back(smoothImageGaussian);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, fuzzySeparated, FUZZYKERNEL, CONVOLUTION);
    vectorMats.push_back(fuzzySeparated);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, smoothSeparated, SMMOTHKERNELGAUSSIAN, CONVOLUTION);
    vectorMats.push_back(smoothSeparated);
    imshowMulti(str, vectorMats);
    Mat image = (Mat_<float>(2, 2) << 1, 2, 3, 4);
    double determinantValue = determinant(image);
    printf("%.2f\n", determinantValue);

    Mat denoisingImage, fuzzyImage, sharpenImage, gaussianImage, gaussianDenoise;
    vector<Mat> images;
    images.push_back(oceanGrayImage);
    gaussionNoise(oceanGrayImage, gaussianImage, 0, 60);
    images.push_back(gaussianImage);
    spatialFilterUsedSeparatedKernel(gaussianImage, fuzzyImage, FUZZYKERNEL, CONVOLUTION);
    images.push_back(fuzzyImage);
    spatialFilterUsedSeparatedKernel(gaussianImage, denoisingImage, DENOISINGkERNELGAUSSIAN, CONVOLUTION);
    images.push_back(denoisingImage);
    GaussianBlur(gaussianImage, gaussianDenoise, Size(7, 7), 3, 0);
    images.push_back(gaussianDenoise);
    imshowMulti(str, images);
    
    // test the limit condition of gaussian filter kernel.
    // you can find the efficient will be similar to the kernel size 6*std if your size is greater than
    // 6*std.
    Mat gaussian3, gaussian7, gaussian13, gaussian5, gaussian6;
    vector<Mat> images;
    images.push_back(oceanGrayImage);
    Mat gaussianKernel3 = getGaussianKernel_(3, 1);
    Mat gaussianKernel5 = getGaussianKernel_(5, 1);
    Mat gaussianKernel7 = getGaussianKernel_(7, 1);
    Mat gaussianKernel13 = getGaussianKernel_(13, 1);
    // Mat gaussianKernel6 = getGaussianKernel_(6, 1);

    spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussian5, gaussianKernel5, CONVOLUTION);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussian7, gaussianKernel7, CONVOLUTION);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussian13, gaussianKernel13, CONVOLUTION);
    // spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussian6, gaussianKernel6, CONVOLUTION);
    images.push_back(gaussian5);
    images.push_back(gaussian7);
    images.push_back(gaussian13);
    imshowMulti(str, images);
    // test the relationship between the size, standard devatition of the kernel and the size of 
    // the image.
    Mat earthImage = imread("../../resources/1500.webp");
    Mat earthGrayImage, faceGrayImage;
    Mat gaussian31, gaussian93;
    cvtColor(earthImage, earthGrayImage, COLOR_BGR2GRAY);
    cvtColor(faceImage, faceGrayImage, COLOR_BGR2GRAY);
    Mat gaussianKernel31 = getGaussianKernel_(3, 1);
    Mat gaussianKernel132 = getGaussianKernel_(13, 2);
    spatialFilterUsedSeparatedKernel(earthGrayImage, gaussian31, gaussianKernel31, CONVOLUTION);
    spatialFilterUsedSeparatedKernel(faceGrayImage, gaussian93, gaussianKernel132, CONVOLUTION);
    vector<Mat> imageTest;
    imageTest.push_back(earthGrayImage); 
    imageTest.push_back(gaussian31); 
    imageTest.push_back(faceGrayImage); 
    imageTest.push_back(gaussian93); 
    imshowMulti(str, imageTest);
    // test reduce the shadow
    Mat shadowImage = imread("../../resources/shadow.webp");
    Mat shadowGrayImage, gaussianShadow, gaussian111;
    uchar *shadowGrayRow, *gaussianShadowRow;
    gaussian111.create(shadowGrayImage.cols, shadowGrayImage.rows, CV_64F);
    cvtColor(shadowImage, shadowGrayImage, COLOR_BGR2GRAY);
    Mat gaussianShadowKernel132 = getGaussianKernel_(181, 30);
    spatialFilterUsedSeparatedKernel(shadowGrayImage, gaussianShadow, gaussianShadowKernel132, CONVOLUTION);
/*     for (size_t i = 0; i < shadowGrayImage.rows; i++)
    {
        shadowGrayRow = shadowGrayImage.ptr<uchar>(i);
        gaussianShadowRow = gaussianShadow.ptr<uchar>(i);
        for (size_t j = 0; j < shadowGrayImage.cols; j++)
        {
            gaussian111.at<double>(i, j) = (double)(shadowGrayRow[j] / gaussianShadowRow[j]);
        }
    }
    linearScaling(gaussian111, outputImage); */

    vector<Mat> imageTestShadow;
    imageTestShadow.push_back(shadowGrayImage); 
    imageTestShadow.push_back(gaussianShadow); 
    imshowMulti(str, imageTestShadow);
    // test the medianFilter function, median filter can reduct the noise and does not increase the degrees of fuzzying.
    // compare the effcient of the median filter and gaussian filter kernel, smooth filter kenel.
    // gaussian filter can reduct the noise, but the efficient is not good and it can just reduct the noise
    // that suitable for the gaussian distribution.
    Mat saltPepperImage, medianFilterImage, smoothFilerImage, gaussianImage71, gaussianImag132;
    Mat gaussianKernel71 = getGaussianKernel_(7, 1);
    Mat gaussianKernel132 = getGaussianKernel_(13, 2);
    saltPepper(oceanGrayImage, saltPepperImage, 200, 3);
    medianFilter(saltPepperImage, medianFilterImage, 5);
    spatialFilterUsedSeparatedKernel(saltPepperImage, smoothFilerImage, SMOOTHKERNELCASSETTE, CONVOLUTION);
    spatialFilterUsedSeparatedKernel(saltPepperImage, gaussianImage71, gaussianKernel71, CONVOLUTION);
    spatialFilterUsedSeparatedKernel(saltPepperImage, gaussianImag132, gaussianKernel132, CONVOLUTION);
    vectorImages.push_back(oceanGrayImage);
    vectorImages.push_back(saltPepperImage);
    vectorImages.push_back(medianFilterImage);
    vectorImages.push_back(smoothFilerImage);
    vectorImages.push_back(gaussianImage71);
    vectorImages.push_back(gaussianImag132);
    imshowMulti(str, vectorImages);
    // test the difference between the four laplacian operator.
    // the param c of former two operators is euqal to -1, the last is 1. notice it.
    Mat laplacianImage, laplacianImage_, laplacianImage__, laplacianImage___, laplacianImage____;
    Mat smoothImage, fuzzyImage, gaussianImage71, gaussianImage132;
    spatialFilterUsedSeparatedKernel(oceanGrayImage, laplacianImage_, SHARPENKERNEL_, CONVOLUTION, true);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, laplacianImage__, SHARPENKERNEL__, CONVOLUTION, true);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, laplacianImage___, SHARPENKERNEL___, CONVOLUTION, true);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, laplacianImage____, SHARPENKERNEL____, CONVOLUTION, true);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussianImage71, GAUSSIANKERNEL71, CONVOLUTION, false);
    spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussianImage132, GAUSSIANKERNEL132, CONVOLUTION, false);
    vectorImages.push_back(oceanGrayImage);
    vectorImages.push_back(laplacianImage_);
    vectorImages.push_back(laplacianImage__);
    vectorImages.push_back(laplacianImage___);
    vectorImages.push_back(gaussianImage71);
    vectorImages.push_back(gaussianImage132);
    imshowMulti(str, vectorImages);
    // test the image of sharpening function used passitation template image.
    // operateTwoMatMultiThread(oceanGrayImage, oceanGrayImage, outputImage, ADD);
    // you can change the param k in the sharpenImageUsedPassivationTemplate function to adjust the
    // efficient of shapening. bigger k, more degrees of sharpening.
    Mat maskImage, passivationImage1, passivationImage2, passivationImage3, passivationImage4;
    sharpenImageUsedPassivationTemplate(oceanGrayImage, passivationImage1, FUZZYKERNEL, 1);
    sharpenImageUsedPassivationTemplate(oceanGrayImage, passivationImage2, FUZZYKERNEL, 2);
    sharpenImageUsedPassivationTemplate(oceanGrayImage, passivationImage3, FUZZYKERNEL, 3);
    sharpenImageUsedPassivationTemplate(oceanGrayImage, passivationImage4, FUZZYKERNEL, -1);
    getMaskImage(oceanGrayImage, maskImage, SMOOTHKERNELCASSETTE);
    vectorImages.push_back(oceanGrayImage);
    vectorImages.push_back(passivationImage1);
    vectorImages.push_back(passivationImage2);
    vectorImages.push_back(passivationImage3);
    vectorImages.push_back(passivationImage4);
    vectorImages.push_back(maskImage);
    imshowMulti(str, vectorImages);
    // test the different efficient that the image of sharpening used template method used different param.
    Mat fuzzyImage, fuzzyGrayImage, sharpenBeauty1, sharpenBeauty2, sharpenBeauty3;
    Mat sharpenBeauty4, sharpenBeauty5;
    fuzzyImage = imread("../../resources/fuzzyImage.webp");
    cvtColor(fuzzyImage, fuzzyGrayImage, COLOR_BGR2GRAY);
    sharpenImageUsedPassivationTemplate(fuzzyGrayImage, sharpenBeauty1, GAUSSIANKERNEL132, 1);
    sharpenImageUsedPassivationTemplate(fuzzyGrayImage, sharpenBeauty2, GAUSSIANKERNEL132, 3);
    sharpenImageUsedPassivationTemplate(fuzzyGrayImage, sharpenBeauty3, GAUSSIANKERNEL132, 5);
    sharpenImageUsedPassivationTemplate(fuzzyGrayImage, sharpenBeauty4, GAUSSIANKERNEL193, 5);
    sharpenImageUsedPassivationTemplate(fuzzyGrayImage, sharpenBeauty5, GAUSSIANKERNEL71, 5);
    vectorImages.push_back(fuzzyGrayImage);
    vectorImages.push_back(sharpenBeauty1);
    vectorImages.push_back(sharpenBeauty2);
    vectorImages.push_back(sharpenBeauty3);
    vectorImages.push_back(sharpenBeauty4);
    vectorImages.push_back(sharpenBeauty5);
    imshowMulti(str, vectorImages);
    // test to strengthen the edge used gradient. sobel operators are the gradient kernel.
    // of course, you can find this function will return the edge image.
    Mat edgeStrengthenImage, sharpenImageBasedonSobel, sharpenImageBasedonLaplacian;
    Mat maskImage, sharpenImageBasedonTemplate, fuzzyImage, gaussianImage;
    Mat darkImages = imread("../../resources/darkImages.webp", 0);
    Mat faceGrayImage;
    cvtColor(faceImage, faceGrayImage, COLOR_BGR2GRAY);
    edgeStrengthenUsedSobelOperator(faceGrayImage, edgeStrengthenImage, SOBELOPERATORGX, SOBELOPERATORGY, 355);
    // sharpenImageUsedSobelOperator(oceanGrayImage, sharpenImageBasedonSobel, SOBELOPERATORGX, SOBELOPERATORGY, 355);
    // spatialFilterUsedSeparatedKernel(oceanGrayImage, sharpenImageBasedonLaplacian, SHARPENKERNEL_, CONVOLUTION, true);
    // spatialFilterUsedSeparatedKernel(oceanGrayImage, outputImage, SMOOTHKERNELCASSETTE, CONVOLUTION, false);
    // spatialFilterUsedSeparatedKernel(oceanGrayImage, fuzzyImage, FUZZYKERNEL, CONVOLUTION, false);
    // spatialFilterUsedSeparatedKernel(oceanGrayImage, gaussianImage, GAUSSIANKERNEL71, CONVOLUTION, false);
    // getMaskImage(oceanGrayImage, maskImage, GAUSSIANKERNEL71);
    // sharpenImageUsedPassivationTemplate(oceanGrayImage, sharpenImageBasedonTemplate, GAUSSIANKERNEL71, 6);
    vectorImages.push_back(faceGrayImage);
    // vectorImages.push_back(sharpenImageBasedonSobel);
    // vectorImages.push_back(sharpenImageBasedonLaplacian);
    // vectorImages.push_back(sharpenImageBasedonTemplate);
    vectorImages.push_back(edgeStrengthenImage);    
    // vectorImages.push_back(sharpenImageBasedonTemplate);
    imshowMulti(str, vectorImages);
    Mat wordImage = imread("../../resources/word.jpg", 0);
    Mat laplacianImage;
    edgeStrengthenUsedSobelOperator(wordImage, laplacianImage, SOBELOPERATORGX, SOBELOPERATORGY, 0);
    vectorImages.push_back(wordImage);
    vectorImages.push_back(laplacianImage);
    imshowMulti(str, vectorImages);
    // test document recognition.
    // Mat wordImage = imread("../../resources/word.jpg", 0);
    // // Mat dialtImage = preprocess(wordImage);
    // wordRegionExtract(wordImage, outputImage);
    // vectorImages.push_back(wordImage);
    // vectorImages.push_back(outputImage);
    // imshowMulti(str, vectorImages);
    // string base64Code = Base64::Mat2Base64(wordImage, ".jpg");
    // Mat image = Base64::Base2Mat(base64Code);
    // imshow("test", image);
    Mat wordImage = imread("../../resources/noteWord.webp");
    ORC orc = ORC();
    Mat dilImage = orc.preProcessing(wordImage);
    vector<Point> biggest = orc.getContours(dilImage, wordImage);
    orc.drawPoints(wordImage, biggest, Scalar(255, 0, 255));
    vector<Point> reorderbiggest = orc.reorderBiggestPoint(biggest);
    Mat wrapImage = orc.getWrap(wordImage, reorderbiggest, 420, 596);
    Mat resizeImage;
    resize(wrapImage, resizeImage, Size(wordImage.cols * 0.3, wordImage.rows * 0.3));
    imshow("test", resizeImage);
    // test the combination function, but the efficient is not well, because the concept about this
    // function used the max area of detected contours as the text recognition region. so it is 
    // not professional. we will try the other method to detect the text region.
    // Mat wordImage = imread("../../resources/document.jpg");
    Mat wordImage = imread("../../resources/paper.jpg");
    ORC orc = ORC();
    Mat dilImage = orc.preProcessing(wordImage);
    Mat resizeDilImage, resizeWordImage;
    resize(dilImage, resizeDilImage, Size(dilImage.cols * 0.3, dilImage.rows * 0.3));
    vector<Point> points = orc.getContours(dilImage, wordImage);
    resize(wordImage, resizeWordImage, Size(dilImage.cols * 0.3, dilImage.rows * 0.3));
    cout << points << endl;

    imshow("123", resizeDilImage);
    imshow("123123", resizeWordImage);
    // imshow("12345", resizeWordImage);
    // Mat documentImage;
    // orc.documentScanned(wordImage, documentImage);
    // imshow("test", wordImage);
    // test the efficient of documentScanned function
    ORC orc = ORC();
    Mat wordImage = imread("../../resources/note.jpg");
    Mat documentImage, resizeWordImage;
    orc.documentScanned(wordImage, documentImage);
    resize(wordImage, resizeWordImage, Size(wordImage.cols * 0.3, wordImage.rows * 0.3));
    cout << wordImage.size() << ", " << documentImage.size() << endl;
    imshow("123", resizeWordImage);
    imshow("test", documentImage);
    #endif
    Mat wordImage = imread("../../resources/paper111.jpg");
    ORC::documentScannedBasedonMinAreaRect_(wordImage, outputImage);
    #endif
    // --------------------return to test the spatial filter-----------------------------





    waitKey(0);
    // notice, you should destroy all the windows you have created at end.
    destroyAllWindows();
    system("pause");
    return 0;
}
