#include "../include/general.h"

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