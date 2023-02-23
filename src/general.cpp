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
    double *returnPointer = NULL;
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


int extractNum(string &ss, char *ch)
{
    const char* c = ss.c_str();
    int amount = 0;
    int i = 0;
    while (c[i] != '\0')
    {
        if (c[i] >= '0' && c[i] <= '9')
        {
            ch[amount] = c[i];
            amount++;
        }
        i++;
    }
    return amount;
}

bool compareVectorString(std::string str1, std::string str2)
{
    char char1[10], char2[10];
    extractNum(str1, char1);
    extractNum(str2, char2);
    // you have get the integer from string, but it is stored use char.
    // you can use strcmp to compare two char pointer. of course, you can use > direct, but it can
    // just compare the first char that first pointer point to. so you'd better use the strcmp function.
    // it will return 1, 0, -1. but we just want to get 0 or 1. only done this you can return bool or false.
    // or you will get error when you return -1. because the return value is bool.
    // notice, you should return one bool value.
    int value = strcmp(char1, char2) < 0 ? 0 : strcmp(char1, char2);
    return value;
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

/**
 * @Author: weiyutao
 * @Date: 2023-02-19 21:50:39
 * @Parameters: string dir. fileNames, filePaths. countImage.
 * @Return: .jpg or .png files. and return the numbers of these files.
 * @Description: return all image files from one directory. notice, it can only read files, can not
 * read directory.
 */
void getImageFileFromDir(const string dirPath, std::vector<cv::String> &imageNames, std::vector<cv::String> &imagePaths, int &countImage)
{
    cv::glob(dirPath, imagePaths);
    imageNames = imagePaths;
    for (size_t i = 0; i < imagePaths.size(); i++)
    {
        if ((imagePaths[i].find(".jpg") != imagePaths[i].npos) || (imagePaths[i].find(".png") != imagePaths[i].npos))
        {
            size_t position = imagePaths[i].find_last_of('\\');
            size_t length = imagePaths[i].find_last_of('.');
            imageNames[i] = imagePaths[i].substr(position + 1, length - position - 1);
            countImage++;
        }
    }
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-16 16:17:14
 * @Parameters: count is aimed to record the number of recurisiving. the current recursiving
 * numbers is the label of current face sample.
 * @Return: 
 * @Description: notice, in order to record the numbers of recursiving, you should pass
 * an int variable from main function and use the reference to pass the parame into the function.
 * only this operation can you define the different label for the different faces.
 * and you should count++ before the recursive code. or you will not get the efficient 
 * what you want. of course, we can read the data and set label from memory directly, 
 * need not to store the txt file in diskdriver, but it is also meaningful to work.
 * because read the label file from the disk dirver is the formal process. it is useful
 * for handle the large project.
 * 
 * then, we should define the formal process that read the label file from the disk drive file
 * and train the face samples based on the file. then predict the dest movie based on
 * the trained model.
 */
void getAllFileFromDirAndCreatTrainData(const string directoryPath, vector<string> &imagePath,\
     const string txtPath, int &count)
{
    DIR *pDir;
    struct dirent *ptr = (struct dirent *)malloc(sizeof(struct dirent));
    struct stat infos;
    ofstream outfile(txtPath, ios::app);
    if (!(pDir = opendir(directoryPath.c_str())))
    {
        sys_error("folder does not exist...");
    }
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            string scanPath = directoryPath + "/" + ptr->d_name;
            if (stat(scanPath.c_str(), &infos) != 0)
            {
                sys_error("scaned file error");
            }
            else if (infos.st_mode & S_IFDIR)
            {
                // DIRECTORY
                count++;
                getAllFileFromDirAndCreatTrainData(scanPath, imagePath, txtPath, count);
            }
            else if (infos.st_mode & S_IFREG)
            {
                // FILE, you should push tha complete path into vector
                // and store the txt file.
                imagePath.push_back(scanPath);
                scanPath += ";";
                scanPath += count;
                outfile << scanPath;
                outfile << endl;
            }
            else
            {
                sys_error("error...");
            }
        }
    }
    sort(imagePath.begin(), imagePath.end());
    closedir(pDir);
    outfile.close();
    freePointer(ptr);
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-21 17:09:53
 * @Parameters: inputImage, one m*m dimension matrix.
 * @Return: 
 * @Description: rotate the matrix, notice it is not the rotation about one image.
 * define a function what can rotate Mat 90 degrees. this function should be suitable for any dimension
 * Mat param. but the one dimension matrix is different from the two dimension matrix, so we will define
 * one extra function that dedicated to rotating the one dimensin matrix 180 degrees.
 */
void rotationMat90(Mat &inputImage)
{
    int n = inputImage.rows;
    if (n == 0)
    {
        return;
    }
    int r = (n >> 1) - 1;
    int c = (n - 1) >> 1;
    for (int i = r; i >= 0; --i)
    {
        for (int j = c; j >= 0; --j)
        {
            swap(inputImage.at<float>(i, j), inputImage.at<float>(j, (n - i - 1)));
            swap(inputImage.at<float>(i, j), inputImage.at<float>((n - i - 1), (n - j - 1)));
            swap(inputImage.at<float>(i, j), inputImage.at<float>((n - j - 1), i));
        }
    }
}

void rotationMatVector(Mat &inputImage, int degrees) 
{
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int halfRows = rows >> 1;
    int halfCols = cols >> 1;
    if (rows != 1 && cols != 1)
    {
        sys_error("you should input one vector...\n");
    }
    if (degrees == NINTY)
    {
        transpose(inputImage.clone(), inputImage);
    }
    else if (degrees == ONEEIGHTZERO)
    {
        for (int i = halfRows; i >= 0; i--)
        {
            for (int j = halfCols; j >= 0; j--)
            {
                swap(inputImage.at<double>(i, j), inputImage.at<double>(rows - i - 1, cols - j - 1));
            }
        }
    }
}

void rotationMat(Mat &inputImage, int degrees) 
{
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    if (rows == 1 || cols == 1)
    {
        rotationMatVector(inputImage, degrees);
        return;
    }
    if (degrees == DEGREES::NINTY)
    {
        rotationMat90(inputImage);
        return;
    }
    if (degrees == DEGREES::ONEEIGHTZERO)
    {
        rotationMat90(inputImage);
        rotationMat90(inputImage);
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-22 12:47:02
 * @Parameters: 
 * @Return: 
 * @Description: 
 */
int getRankFromMat(Mat &inputImage) 
{
    if ((inputImage.channels() != 1) || inputImage.empty())
    {
        sys_error("the inputImage is empty? please pass the one channel image...");
    }
    Eigen::MatrixXd temp;
    cv::cv2eigen(inputImage, temp);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(temp);
    int rank = svd.rank();
    return rank;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-22 12:45:56
 * @Parameters: w, one Mat that can be separated.
 * @Return: 
 * @Description: you should ensure the Mat w can be separated. or you will get error.
 * the thought to separate one m order matrix  w(m, m) is.
 * step1, find any nonzero element. just define it as an variable E(Scalar).
 * step2, find the row of E in the matrix w. defined it as the column vector w2(1, m)
 * step3, find the column of E in the matrix w, defined it as the vector w1(m, 1).
 * step4, w2 /= E
 * step5, w = w1 @ w2.
 */
void separateKernel(Mat &w, Mat &w1, Mat &w2) 
{
    int rows = w.rows;
    int cols = w.cols;
    double *wRow;
    for (int i = 0; i < rows; i++)
    {
        wRow = w.ptr<double>(i);
        for (int j = 0; j < cols; j++)
        {
            if (wRow[j] != 0)
            {
                w2 = w.rowRange(i, i + 1).clone();
                w2 /=  wRow[j];
                w1 = w.colRange(j, j + 1).clone();
                return;
            }
        }
    }
}



/**
 * @Author: weiyutao
 * @Date: 2023-02-22 18:13:10
 * @Parameters: 
 * @Return: 
 * @Description: the function get GaussianKernel, of course, you can also use the official function
 * getGaussianKernel.
 */
Mat getGaussianKernel_(const int size = 3, const double sigma = 1.0)
{
    double **gaus = new double *[size];
    for (int i = 0; i < size; i++)
    {
        gaus[i] = new double[size];
    }
    Mat Kernel(size, size, CV_64FC1, Scalar(0));
    const double PI = 4.0 * atan(1.0);
    int center = size / 2;
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            gaus[i][j] = (1 / (2 * PI * sigma * sigma)) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
            sum += gaus[i][j];
        }
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            gaus[i][j] /= sum;
            Kernel.at<double>(i, j) = gaus[i][j];
        }
    }
    return Kernel;
}
