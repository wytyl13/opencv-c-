#include "../include/bitOperation.h"

void bitWise(Mat inputImage, Mat &outputImage) 
{
    // first, you should define a rect object.
    Rect rect = Rect(Point(100, 50), Point(500, 250));
/*     cout << rect.area() << endl;
    cout << rect.tl() << endl;
    cout << rect.br() << endl;
    cout << rect.height <<endl;
    cout << rect.width << endl; */
    // notice the difference between Size and the other class, just like Point.
    // the first param in Size is width, the second is height.
    // the first param of  other object is x, the second param is y.
    // cout << rect.size() << endl;
    // rect is a rectangular, left top point is 100, 50; right bottom point is 500, 250;
    Mat image = Mat::ones(inputImage.size(), inputImage.type());
    // set value 255 to each element for the mat. you can use setTo, copyTo or clone function.
    image.setTo(255);
    // cout << image.size() << endl;
    // cout << (int)image.ptr<uchar>(0)[1] << endl;
    // bit wise and operation, ^, the binary of dec 255 is 1111 1111.
    // any binary ^ 1111 1111 is equal to itself.
    // we can set the region used 0 value where we want. then, we will get the blank region. 
    // you can use fillconvxpolly function to set the region of the inputimage used a scalar.
    // you can also use fillPoly function to set the value for a polygon if the polygon is involved some points.
    // but how to fill the value used a rect object?    
    // you can use & | ~ to caluclate two Rect.
    // notice, fillPoly and fillConvxPoly are all used point to fill.
    // if you want to fill a Rect region in one image. you can define four point based on the Rect.
    // you can use vector to store the four points.
    vector<Point> vectorPoints;
    Point tl = rect.tl();
    Point br = rect.br();
    vectorPoints.push_back(tl);
    vectorPoints.push_back(Point(br.x, tl.y));
    vectorPoints.push_back(br);
    vectorPoints.push_back(Point(tl.x, br.y));
    // fill the region based on the four point that get them used the Rect.
    fillPoly(image, vectorPoints, Scalar(0, 0, 0));
    bitwise_and(inputImage, image, outputImage);
}