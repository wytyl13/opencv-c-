#include "../include/featureInImage.h"

/**
 * @Author: weiyutao
 * @Date: 2023-02-14 11:42:26
 * @Parameters: 
 * @Return: 
 * @Description: feature extract involved feature detection and feature describe.
 * so it means the feature extract will describe the feature information based on the
 * detected features. how to deep understand it? just like we used traget angle as
 * characteristics. then, the purpose of feature extract is to find angle in one image
 * or some region in one image. and you should describe some attribution of thses detected angles.
 * just like the direction, position and so on. we can image some other attribution in digital
 * image process, just like resize, scale, rotation, position and so on, you can consider the feature
 * of one image is an extra attribution, and it is independent of the other attribution.
 * it means the features of one image should not be changed with the rotation, resize, position and
 * scale of the image. before contracting features, you should pretreatment the image as far as possible.
 * for example, you can use histograme equalization or histogram matching method to enhance
 * the contrast of the image. the purpose of pretreatment image is to improving the accuracy of
 * feature extraction.
 * 
 * how to deep understand the independency of the features.
 * simply to see it, just like the different rotation of the same image.
 * the feature that you detected will be different, because the rotation is different.
 * but the area of the features in different rotation image is similar. so we can conclude that
 * the different describe will be different when you transformed some basic operation based on the original image.
 * so we can introduction two meaning for the attribution feature for one image. the feature attribution in
 * one image will involved two meaning, one is constant, one is convariance, constant means the different
 * image feature describe is similar, just like the attribution area of feature will constant for two different rotation image.
 * and the attribution area of feature will be convarince for two different size image. so how to improve the accuracy
 * for the feature detected in image? the constant attribution describe will be simple to judge it. what we need to focus on
 * is how to find the regular about the convariance. it means the best method is to normalize the convariance to
 * relevant invariance as far as possible. just like you can scale all feature used the scale rate, then, the erea describe will
 * be invariance for the different size image.
 * 
 * then, we have introduced the invariance and convariance, then, we will introduce the local and global reature.
 * but the problem is one feature may both be local and global feature. before defining that attribution, we should
 * introduce the memeber and set. the set is consisted of multiply members. it will be a local feature
 * if it is used in a member. on the contrary, it will be a global feature if it is used in a set. what is member and
 * what is set? it will be depend on the specific application. for example, the area feature in one bottle, of course, the 
 * area in the bottle is the liquid, just like there is a production line, the application is to calculate the
 * total area of liquid in all bottle what through the production line during 10:00 to 12:00 two hour based on the picture. then, you can
 * then, the area feature for each picture will be local feature, and the total area for all picture will be
 * global feature, it is equal to each picture area * numbers, but the area for picture will be global feature if you want to
 * calculate liquid area in one picture that a certain point through the production line. so the local or global feature attribution
 * is determined by the specific application. then, how to describe these feature? they are described by
 * gray value, just like RGB image, one element is consisted of three gray value at least, each gray value is corresponding to
 * the describe of red, green and blue. of course, you can add the describe in the element, just like n dimension.
 * it means the image has n feature. it is meaningless to generate feature data besed on one image, but what is generally done
 * is used the feature data to implement some advanced application. just like model search.
 * 
 * similaly, feature can be also divided into three class.
 * boundary feature, region feature and the overall image feature.
 * why boundary feature? just like it is meaningless for the length of one image. but it is meaningful for
 * the boundary lenght of one image. the subsequent discussion will be based on boundary and region.
 * 
 * then, we will implement the feature extract algorithm, but we should understand the boundary tracking 
 * algorithm.
 */
void featureExtract(Mat &inputImage) 
{
    return;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-14 11:43:02
 * @Parameters: 
 * @Return: 
 * @Description: 
 */
void featureDetect(Mat &inputImage) 
{
    return;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-14 14:30:07
 * @Parameters: 
 * @Return: 
 * @Description: the boundary tracking algorithm. it can also be named as cellular automata theory.
 * how to understand the moore algorithm. it can find the boundary in the image. you can image a simple 
 * scenery. we will define two symbol to represent the element. just like b is represented the boundary.
 * c will be a scanner what scaned 8 neighbohood of b element. n_i, i is from 1 to 8. the started scan position 
 * is what we should focus on. and the first position of scanning the element is not zero.
 * then, we shoud define the detailed step about moore algorithm.
 * 
 * step1, define the position of b0 and c0. the position of b0 is the first position of nonzero element
 * at leftUpper of the image. c0 is the west of the first element of b0.
 * 
 * step2, started to scan from the position of c0, scan the eight neighborhood of b, n1 = the gray value of the position c.
 * if nk != 1, the position of b1 is equal to the position of nk. nk is the first nonzero position in the eight neighborhood of b0.
 * update the position of c1 is equal to the position of nk-1. ...
 * 
 * step3, over and over again util scan end. it menas you should end if the next position of b is equal to the position of b0.
 * we will simple this problem, image one binary gray value image as follow.
 * 0 0 0 0 0 0 0
 * 0 0 1 1 1 1 0
 * 0 1 0 0 1 0 0
 * 0 0 1 0 1 0 0
 * 0 1 0 0 1 0 0
 * 0 1 1 1 1 0 0
 * 0 0 0 0 0 0 0
 * the original coordinate is (0, 0)
 * the coordinate of b0 is (1, 2), c0 is (1, 1)
 * started to scan, n1 = (1, 1) = 0, n2 = (0, 2) = 0, n3 = (0, 3) = 0, n4 = (1, 3) = 1. n4 ....
 * then, b1 = the first nonzero gray value coordinate = the position of n4 = (1, 3)
 * c1 = the former of the first nonzero gray value coordinate = the position of n3 = (0, 3)
 * 
 * over and over again...
 * b2 = (1, 4), c2 = (0, 4), b3 = (1, 5), c3 = (0, 5), b4 = (2, 4), c4 = (2, 5)
 * b5 = (3, 4), c5 = (3, 5), b6 = (4, 4), c6 = (4, 5), b7 = (5, 4), c7 = (5, 5)
 * b8 = (5, 3), c8 = (6, 3), b9 = (5, 2), c9 = (6, 2), b10 = (5, 1), c10 = (6, 1)
 * b11 = (4, 1), c11 = (4, 0), b12 = (3, 2), c12 = (3, 1), b13 = (2, 1), c13 = (3, 1)
 * b14 = (1, 2) = the position of b0, so the process end. boundary is detected successfully.
 * draw line based on all point in b will be the boundary in this picture.
 * notice scaned to clockwise.
 * 
 * moore boundary tracking algorithm can handle all complex problem
 * it will handle the boundary with the branch, the intersected boundary with a processing.
 * it will handle multi boundary in one picture with multi processing. it means you can not find multi boundary
 * in one picture by one processing. the width of boundary will be 1 element. you can also find the inner region of 
 * based on the detected boundary. of course, you can also counterclockwise to scan the eight neiborhood.
 * 
 * we have got the boundary, then how to connect each boundary element? the original method is freeman chain code.
 * what is chain code? just like freeman chain code, it involved four direction and 8 direction.
 * just like a case as follow
 *   1 1                
 * 1     1
 * 1 1 1 1
 * 4 direction chain code will be a 凸 shape.
 * 8 direction chain code will be a shape as follow.
 *  __
 * /  \
 *|____|
 * the 8 direction chain code can show the more accurate details than 4 direction.
 * of course, they are all based on the richer network.
 * just like the feature for one image, you can consider it as all element for one image.
 * one picture has n features, it also has n elements.
 */   
void boundaryTrackingUsedMoore(Mat &inputImage) 
{
    return;
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-15 10:37:33
 * @Parameters: 
 * @Return: 
 * @Description: the hoffmain chain code function what is dedicated to drawing all the features
 * in sets.
 */
void hoffmanChainCode() 
{
    return;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-14 19:25:45
 * @Parameters: inputImage, one RGB picture. outputImage, the outgoing param.
 * @Return: 
 * @Description: this function will use the official method to track the boundary. we will use 
 * findCountours function. before using this function, you should pretreatment the picture first.
 * you should read a gray picture used imread function directly if you want to enhance your efficient
 * of process. this function will detect all the feature and draw the feature points add
 * some limit conditions. and this function will mark the center of each detected object, and
 * mark the direction and angle used PCA. then, we will define another function 
 * faceFeatureDetect what is dedicated to extracting all the feature in the face.
 */
void boundaryTrackUsedOfficial(Mat &inputImage, Mat &outputImage) 
{
    Mat grayImage, contrastImage, binaryImage, sharpenImage;
    vector<Vec4i> hireachy;
    vector<vector<Point>> contours;
    outputImage = inputImage.clone();
    if (inputImage.channels() != 1)
    {
        cvtColor(inputImage, grayImage, COLOR_RGB2GRAY);
    }
    else
    {
        grayImage = inputImage.clone();
    }
    // it is not suitable for enhancing the contrast of one binary image.
    // equalizeHist(grayImage, contrastImage);
    threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // the efficient of bianry image is not good, because some fuzzy line is not recognized.
    // so we will consider sharpen the image first. we have defined these filter transformation
    // function in spatial filter file, it is named officialFilterTest function.
    // then, we have found that the binary image is not working good if you want to
    // enhance the contrast of the picture. so the pretreatment for binary image is not suitable.
    // officialFilterTest(binaryImage, sharpenImage, SFD::SHARPEN);
    // this official function findContours involved feature detected, you will get all the feature sets
    // what is stored in contours. we will implement the code about feature detected what means boundary
    // tracked in the above function boundaryTrackingUsedMoore. the moore algorithm is a boundary tracked
    // algorithm. this is the first step about feature extraction.
    // the second step is draw all the features. we will use official function drawContours what is a
    // special function to draw the feature sets. it used some chain code method. just like hoffman chain code.
    // we will implement the hoffman chain code in the above function.
    findContours(binaryImage, contours, hireachy, RETR_LIST, CHAIN_APPROX_NONE);

    // double *list = (double *)malloc(sizeof(double) * 2);
    double *list = NULL;

    for (long long unsigned i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        double length = arcLength(contours[i], true);
        if (area > 1e5 || area < 1e2)
        {
            continue;
        }
        printf("the current object area is %lf\n", area);
        printf("the current object length is %lf\n", length);
        drawContours(outputImage, contours, i, Scalar(0, 0, 255), 2, 8);
        // mark the center of each contour. we will use PCA algorithm to implement it.
        // why use PCA in the application of feature extrantion? you can get the direction
        // what can remain the max information of each contour what is a feature in this image.
        // and you can get the center of each feature used feature vector what you can calculate it used
        // the convariance matrix, because the center must be on the feature vector.
        // the method by calculating feature vector and return the max k vector based on the convariance matrix
        // is PCA, we can reduce the dimension of inputImage from n to k and maximize to remain the original information.
        // the max k feature vector is the max information direction. and the center of the inputImage
        // must be one the feature vector. then, we should define a PCA function to handle each contour 
        // we have found used findContours function.
        calcPCAOrientationUsedOfficial(contours[i], outputImage);
        list = (double *)calculateAttributionBasedOnFeaturePoints(contours[i], ACS::CENTER);
        printOneArrayPointer(list, 2);

    }
    freePointer(list);
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-15 11:04:22
 * @Parameters: 
 * @Return: 
 * @Description: contour what is stored all the feature point in one feature. just like a object in
 * one image. we will describe it used many feature point, then draw all the feature point used chain code.
 * then, cycle it we will get all objects in the image. so each contour stored all point what is showed
 * used coordinate. so we will calculate based on the coordinates.
 */
double calcPCAOrientationUsedOfficial(vector<Point> &pts, Mat &outputImage) 
{
    int size = pts.size();
    // 64F means 64 bits double, C1 mean one channel.
    // why should transform from vector<Point> to Mat? because the PCA function 
    // can just accept the Mat data type.
    Mat ptsCoordinatesMat = Mat(size, 2, CV_64FC1);
    for (int i = 0; i < size; i++)
    {
        ptsCoordinatesMat.at<double>(i, 0) = pts[i].x;
        ptsCoordinatesMat.at<double>(i, 1) = pts[i].y;
    }

    // init the pca class. the data is a matrix that consists of all the feature point of one detected object
    // in the inputImage. the concept is calculate the convariance matrix of the ptsCoordinatesMat, 
    // then calculate the featureValues and featuerVectors based on the convariance matrix.
    // the featureValues and featureVectors used eigenvalues and eigenvectors these two attribution to show.
    // data in eigenvalues and eigenvectors has sorted, you should index 0 data from them. the vector of index 0
    // is the direction of the object what is also the line you should draw in the image.
    // the direction described used vector what is a vector that must throght the original of the
    // descartes coordinate system. the difference is the original point for each detected object is
    // the center of the detected object. you can calculate it used the moments or the attribution mean
    // of pca class based on the featuer points set of one detected object.
    PCA pca(ptsCoordinatesMat, Mat(), PCA::DATA_AS_ROW);

    // then, you should consider the concept about the pca find the center of one object based on
    // the feature points set of one object in one image. you can also use moments function in opencv
    // to get the center of one objetc what is displayed used feature points what you can also considered
    // it as a outline. of course, you can also consider the calculation method about the length, area
    // center, and other attribution of the contours what is the coordinates set, you can show it used
    // vector or Mat. it can show the information about one object in one image.
    Point center = Point((int)pca.mean.at<double>(0, 0), (int)pca.mean.at<double>(0, 1));
    cout << "the center is: " << center << endl;
    circle(outputImage, center, 2, Scalar(0, 255, 0), 2, 8, 0);

    // calculate the feature value and feature vector based on the convariance matrix.
    // we will use the eigenvalues and eigenvectors attribution in PCA class in opencv
    //  to calculate them. we will get the max 2 feature vectors and feature value.
    // we have passed the coordinates set what is the original matrix, we will get the convariance
    // matrix based on it, then, we will use PCA method to get all attribution of the convariance
    // matrix.

    vector<Point2d> featureVectors(2);
    vector<double> featureValues(2);
    for (int i = 0; i < 2; i++)
    {
        featureValues[i] = pca.eigenvalues.at<double>(i, 0);
        featureVectors[i] = Point2d(pca.eigenvectors.at<double>(i, 0), pca.eigenvectors.at<double>(i, 1));
    }
    Point point1 = center + 0.02 * Point((int)(featureVectors[0].x * featureValues[0]), (int)(featureVectors[0].y * featureValues[0]));
    Point point2 = center - 0.05 * Point((int)(featureVectors[1].x * featureValues[1]), (int)(featureVectors[1].y * featureValues[1]));

    line(outputImage, center, point1, Scalar(255, 0, 0), 2, 8, 0);
    line(outputImage, center, point2, Scalar(255, 0, 0), 2, 8, 0);

    double angle = atan2(featureVectors[0].y, featureVectors[0].x);
    printf("angle: %.2f\n", 180 * (angle / CV_PI));
    return angle;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-15 16:49:47
 * @Parameters: 
 * @Return: 
 * @Description: this function will detect all features in the image. and you can judge
 * the object based on these features. just like you can define a face classifier, 
 * it has defined all feature what should be involved in the face shape. so if the
 * features that you detected are suitable for it, you will get it is a face. then 
 * you can get the minimize boundary for the face, then you can draw the boundary 
 * based on the boundary point. so the feature detected is the basic for the face detected.
 * you can consider the face detected is a super application that is basised by the
 * feature detected. of course, the face detected need not use the chain code to 
 * draw the boundary line. the boundary line just need to be used when you want to show
 * all the features in your image. face detected just need to rectangular the minimize
 * face region. just like the special face detected, any object detected is similar
 * to it. the premise is you should define the classifier about the object, just like dog,
 * cat, desk, banana, apple and so on. they are insisted of many features. you can judge it
 * based on the feature numbers and distance between two features.
 * 
 * then, you should define the object detected classifier if you get an badly
 * efficient used the existing classifier. we will subsequent learn how to define the
 * object classifier.
 * 
 * of course, the boundary detected function are also the super application for
 * feature detected in opencv, because we should get all the features used some
 * existing algorithm. we will define them in above function featureDetectUsedOfficial.
 * 
 * so you should understand the relationship about feature detect and boundary detect.
 * the former is the basic, the last is draw a closed shape that consisted of the nearby
 * feature points on the basis of the fomer operation.
 * 
 * you should know, the feature point is all the point that it should be defined
 * as feature point based on some algorithm, just like angle point, or other algorithm.
 * the boundary is an object that consists of the nearby feature points. notice, it is not
 * the simple boundary tracked. the boundary tacked is the simple and rough algorithm.
 * you can consider the boundary is one type for the boundary detected.
 * in order to deep understand the feature detect and boundary detect, we should learn
 * some deature detected algorithm.
 */
void boundaryDetectUsedOfficial(Mat &inputImage, Mat &outputImage) 
{
    Mat grayImage, contrastImage, binaryImage;
    vector<Vec4i> hireachy;
    vector<vector<Point>> contours;
    outputImage = inputImage.clone();
    if (inputImage.channels() != 1)
    {
        cvtColor(inputImage, grayImage, COLOR_RGB2GRAY);
    }
    else
    {
        grayImage = inputImage.clone();
    }
    threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
    findContours(binaryImage, contours, hireachy, RETR_LIST, CHAIN_APPROX_NONE);
    for (long long unsigned i = 0; i < contours.size(); i++)
    {
        drawContours(outputImage, contours, i, Scalar(255, 255, 0), 1, 8); 
    }   
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-15 17:12:36
 * @Parameters: 
 * @Return: 
 * @Description: you can find that the findContours function used the class
 * Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(); this class
 * is a fast algorithm to detect all features in opencv. findContours is the super
 * applicaiton about it. 
 * 
 */
void featureDetectUsedOfficial(Mat &inputImage, Mat &outputImage) 
{
    Mat grayImage;
    Mat KeyPointImage = inputImage.clone();
    if (inputImage.channels() != 1)
    {
        cvtColor(inputImage, grayImage, COLOR_RGB2GRAY);
    }
    else
    {
        grayImage = inputImage.clone();
    }
    // you should use KeyPoint class to store the features.
    vector<KeyPoint> detectKeyPoint;
    Ptr<FastFeatureDetector> fast = FastFeatureDetector::create();
    fast->detect(grayImage, detectKeyPoint);
    drawKeypoints(KeyPointImage, detectKeyPoint, outputImage, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // drawKeypoints(outputImage, detectKeyPoint, KeyPointImage2, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
}