#include "../include/noise.h"

/**
 * @Author: weiyutao
 * @Date: 2023-01-09 11:38:20
 * @Parameters: srcImage, dstImage, size, count.
 * @Return: the noise image.
 * @Description: this function will create image arithmetic method. the application is noise reduction, 
 * compare and so on. we should learn how to add noise to an image first. involved salt and pepper noise, 
 * gaussian noise and so on. we will define the salt and pepper noise first
 * 
 * 1 generate two randn, x and y, to show a coordinate in one image.
 * 2 generate a randn, 0 or 1, to show it is salt(white) or pepper(black) in the coordinate
 * 3 the param should also involved the size and numbers for the salt and pepper.
 * 4 this function is dedicated to the gray image, it is not suitable for color image.
 * why reference dstImage? to reduce the consumption of the memeory. reference will not take up the extra memory
 * for the parameter dstImage.
 * 
 * then, we will define the gaussian noise method. the difference between gaussion noise and saltPepper noise is 
 * the former should generate the same size like the original image based on the gaussion distribution, the param for gaussion
 * involved mean and var. then you should add the gaussion array with the original image. the result is the the image with gaussion noise.
 * the saltPepper noise is set the element number used 255(white) or 0(black). the element coordinated you should use rand method to get, 
 * and the param for saltPepper involved size and count. the count is the numbers of the white and black, the size is the element size of the white and black
 * then, the param white or black is the binary rand number.
 * the rand method in c++ is rand(), it will return the random unsigned int numbers. but if you just used it, you will get the same
 * random int numbers every time, you should add the code srand((unsigned)time(NULL)) to set the rand seed based on the current time,
 * you should program this code in main function not in one function you will use the rand(), or it will be invalid.
 */
void saltPepper(Mat srcImage, Mat &dstImage, int count, int size) 
{
    dstImage.create(srcImage.size(), srcImage.type());
    dstImage = srcImage.clone();
    int x, y, noiseType;

    while (count)
    {
        x = rand() % (srcImage.rows - size + 1);
        y = rand() % (srcImage.cols - size + 1);
        noiseType = rand() % 2; // 0 or 1.
        if (noiseType)
        {
            // noiseType=1, salt, white
            // you should set the element numbers equal to 255 white based on the size
            // if size == 1; it means you should set 0*0 size, it means you should just set the current
            // postion coordinate number equal to 255.
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dstImage.at<uchar>(x + i, y + j) = 255;
                }
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    dstImage.at<uchar>(x + i, y + j) = 0;
                }
            }
        }
        count --;
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-09 13:06:19
 * @Parameters: 
 * @Return: 
 * @Description: gaussion distribution, you should define the param mean and var. gaussion distribution can be also named 
 * normal distribution. you can use numpy in python to generate it, you can also use Mat in opencv to generate it.
 * you can also use this function to add noise for color image. because the zeros and fill method is general for all imgae.
 * involved gray and color image.
 */
void gaussionNoise(Mat srcImage, Mat &dstImage, float mean = 0, float var = 50) 
{
    dstImage = srcImage.clone();

    // two method. you can use randn in opencv. used Scalar data type pass the mean and var.
    // the second method, you can use the fill function. of course, you can init all elment as 0 used zeros function in Mat
    // you can also define a Mat type variable used Mat directly. this function is general for all type image.

    // you can also use randn function that opencv provided.
    Mat noise(dstImage.rows, dstImage.cols, CV_8UC1);
    randn(noise, Scalar::all(mean), Scalar::all(var));
    dstImage += noise;
    // randn(dstImage, Scalar::all(mean), Scalar::all(var)); this function randn can also generate a gaussion image
    // the size is same with the dstImage and the mean is mean, the var is var.
    // Mat noise = Mat::zeros(dstImage.rows, dstImage.cols, dstImage.type());
    // RNG rng;
    // rng.fill(noise, RNG::NORMAL, mean, var);
    // dstImage += noise;
}
