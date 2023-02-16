/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-02-15 17:49:44
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-02-15 17:49:44
 * @Description: we have learned all the basic knowledge and implement them
 * used official function in opencv. then, we will learn some super application
 * about face detect and face recognition. the face recognition will use 
 * dlib module. and the face detected will use opencv. we should install the dlib
 * first. the install steps is similar to opencv installed. the different is you should
 * mkdir build and install directory in the source directory for dlib. configure and generate
 * the makefiles in build directory,  cd build and make install the dlib into install directory.
 * because dlib will install all the files into the system path, so you should open
 * the terminal used manager permission.
 * -- Up-to-date: C:/Program Files (x86)/dlib_project/include
 * -- Up-to-date: C:/Program Files (x86)/dlib_project/lib
 * the former path is include, the second path is lib.
 * 
 * it is the default install path above, of course, you can also define the customer install path
 * you should add param DESTDIR=the path where you want to install.
 * make install DESTDIR=C:\Users\weiyutao\opt\dlib-19.24\dlib-19.24\install
 * you can find the compiler speed is very slowly if your program used static library and 
 * no matter where location that the path of static library is.
***********************************************************************/
#include "../include/faceApplication.h"


int g_pos = 0;
VideoCapture video;



/**
 * @Author: weiyutao
 * @Date: 2023-02-12 19:58:40
 * @Parameters: inputImage, a original image. 
 * @Return: outputImage, the outgoing param, rectangle the faces on the original image.
 * @Description: we will use the model haarcascade_frontalface_alt2.xml file to implement
 * the detection about face. you should passone original image. then transform it involved gray transformation
 * and histogram equalization transformation and stored it used a temp Mat variable, then draw all the faces used rectangle
 * on the original image.
 */
void faceDetectImage(Mat &inputImage, Mat &outputImage, CascadeClassifier &face_cascade, CascadeClassifier &eye_cascade) 
{
    outputImage = inputImage.clone();
    Mat grayImage;
    cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
    equalizeHist(grayImage, grayImage);
    if (face_cascade.empty())
    {
        face_cascade.load(FACEDETECTMODEL1);
    }
    if (eye_cascade.empty())
    {
        eye_cascade.load(EYEDETECTMODEL);
    }
    vector<Rect> faces;
    vector<Rect> eyes;
    face_cascade.detectMultiScale(grayImage, faces, 1.1, 2, 0, Size(20, 20));
    eye_cascade.detectMultiScale(grayImage, eyes, 1.1, 2, 0, Size(20, 20));
    for (long unsigned int i = 0; i < faces.size(); i++)
    {
        Point upperLeft(faces[i].x, faces[i].y);
        Point lowRight(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        rectangle(outputImage, upperLeft, lowRight, Scalar(0, 0, 255), 2, 8, 0);
        Mat face_ = grayImage(faces[i]);
        eye_cascade.detectMultiScale(face_, eyes, 1.2, 2, 0, Size(30, 30));
        for (size_t j = 0; j < eyes.size(); j++)
        {
            // notice, the eye center is the position that in original image what is inputImage.
            Point centerEye(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(outputImage, centerEye, radius, Scalar(0, 0, 255), 4, 8, 0);
        }
    }
}

void func(int, void *) 
{
    video.set(CAP_PROP_POS_FRAMES, g_pos);
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-12 21:01:23
 * @Parameters: 
 * @Return: 
 * @Description: detect face for a movie, you should pass a movie path, then, this function
 * will detect all face information and show the movie with face rectangle.
 * the video will store used VideoCapture data type.
 */
void faceDetectMovie(const string windowName, const string path, CascadeClassifier &face_cascade, CascadeClassifier &eye_cascade) 
{
    video.open(path);
    double fps;
    int width, height, total_frame;
    if (video.isOpened())
    {
        fps = video.get(CAP_PROP_FPS);
        width = video.get(CAP_PROP_FRAME_WIDTH);
        height = video.get(CAP_PROP_FRAME_HEIGHT);
        total_frame = video.get(CAP_PROP_FRAME_COUNT);
        cout << "the width of picture in movie is " << width << endl;
        cout << "the height of picture in movie is " << height << endl;
        cout << "the total frame rate of movie is " << fps << endl;
        cout << "the total frams of movie is " << total_frame << endl;
    }
    else
    {
        cerr << "read movie error!" << endl;
    }
    int frame_count = video.get(CAP_PROP_FRAME_COUNT);
    Mat frame, outputImage;
    namedWindow(windowName, WINDOW_AUTOSIZE);
    createTrackbar("frame", windowName, &g_pos, frame_count, func);
    int position = getTrackbarPos("frame", windowName);
    while (video.read(frame))
    {
        
        if (frame.empty())
        {
            cerr << "error, read frame from movie error or movie has read end!" << endl;
            break;
        }
        // faceDetectImage(frame, outputImage, face_cascade);
        // set the keyboard reply. the key variable will store the keyboard input information.
        int key = waitKeyEx(30);
        if (key == 27)
        {
            break; // esc
        }
        if (key == 32)
        {
            waitKey(0); // spcace
        }
        if (key == 2424832)
        {
            g_pos -= 30;
            video.set(CAP_PROP_POS_FRAMES, g_pos);
            frame_count = video.get(CAP_PROP_FRAME_COUNT);
            createTrackbar("frame", windowName, &g_pos, frame_count, func);
            setTrackbarPos("frame", windowName, position);
        }
        if (key == 2555904)
        {
            g_pos += 30;
            video.set(CAP_PROP_POS_FRAMES, g_pos);
            frame_count = video.get(CAP_PROP_FRAME_COUNT);
            createTrackbar("frame", windowName, &g_pos, frame_count, func);
            setTrackbarPos("frame", windowName, position);
        }
        if (getWindowProperty(windowName, WND_PROP_AUTOSIZE) != 1)
        {
            break;
        }
        g_pos = video.get(CAP_PROP_POS_FRAMES);
        // you should resize the frame first to reduce the working. then, your process will run smooth.
        resize(frame, frame, Size(width / 2, height / 2));
        faceDetectImage(frame, outputImage, face_cascade, eye_cascade);
        imshow(windowName, outputImage);
        // flip(outputImage, outputImage, 1);
        // imshow(windowName, frame);
        // notice, you should consider the waitkey function here.
        // waitkey(0) means make the window wait forever util press the keyboard.
        // this unit for waitkey is millisecond level. waitkey(1000) means wait 1000 ms.
        // then, when you want to show one image, you should code waitkey(0) after the imshow
        // function. then how to do if you want to show the movie? read each frame of the movie first
        // used while(1), then, the process will wait forever if you code waitkey(0), so you should
        // code waitkey(fps), fps is the frame per second, 
        // 
        // waitKey(fps);
        // CLOSE THE WINDOWS BY judging the variable WND_PROP_AUTOSIZE is euqal to 1?
        // if this variable is equal to 1, it measn you have not clicked the close option in window.

    }
    video.release();
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-13 23:18:38
 * @Parameters: 
 * @Return: 
 * @Description: then, we will learn about face recognition algorithm.
 * the more rich color will bring heavy burden to the performance of the algorithm, 
 * sometimes, we can get the same efficient used  less color number. so reduce the color
 * numbers is a basic operation before coding the algorithm. about image recognition problem,
 * our purpose is efficient. so you can use any method to handle this problem. generally, we used
 * traditional algorithm, just like mean gray value and histogram matching. these two method is
 * efficient. we can use them to filter many image. then, for some special image, just like the same
 * content, the different size, the different background, the different shape and position
 * and so on. so in this case, mean gray value and histogram matching will not work. then, we can 
 * get the feature points in this picture first used the tranditional function. then,
 * compare the number of similar feature points, if the similar degree of image feature points
 * for these two picture is greater than our exprected, we will think they are similar images.
 * just like face recognition, it is comparing the similar degree of face. so the face recognition
 * is a special case for image recognition. then, we will learn about face recognition first.
 * about face recognition, you can find the the face in one picture, then compare these two face.
 * you can find all the feature in these two faces, and then compare the similar degress of these two faces.
 */
void faceRecognition(Mat &image1, Mat &image2) 
{
    int64 t1, t2;
    double tkpt, tdes, tmatch_bf, tmatch_knn;
 
    // 1. 读取图片
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
 
    cv::Ptr<cv::SiftFeatureDetector> sift = cv::SiftFeatureDetector::create();
    // 2. 计算特征点
    t1 = cv::getTickCount();
    sift->detect(image1, keypoints1);
    t2 = cv::getTickCount();
    tkpt = 1000.0*(t2-t1) / cv::getTickFrequency();
    sift->detect(image2, keypoints2);
 
 
    // 3. 计算特征描述符
    cv::Mat descriptors1, descriptors2;
    t1 = cv::getTickCount();
    sift->compute(image1, keypoints1, descriptors1);
    t2 = cv::getTickCount();
    tdes = 1000.0*(t2-t1) / cv::getTickFrequency();
    sift->compute(image2, keypoints2, descriptors2);
 
 
    // 4. 特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    // cv::BFMatcher matcher(cv::NORM_L2);
 
    // (1) 直接暴力匹配
    std::vector<cv::DMatch> matches;
    t1 = cv::getTickCount();
    matcher->match(descriptors1, descriptors2, matches);
    t2 = cv::getTickCount();
    tmatch_bf = 1000.0*(t2-t1) / cv::getTickFrequency();
    // 画匹配图
    cv::Mat img_matches_bf;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, img_matches_bf);
    imshow("bf_matches", img_matches_bf);
 
    // (2) KNN-NNDR匹配法
    std::vector<std::vector<cv::DMatch> > knn_matches;
    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    t1 = cv::getTickCount();
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2);
    for (auto & knn_matche : knn_matches) {
        if (knn_matche[0].distance < ratio_thresh * knn_matche[1].distance) {
            good_matches.push_back(knn_matche[0]);
        }
    }
    t2 = cv::getTickCount();
    tmatch_knn = 1000.0*(t2-t1) / cv::getTickFrequency();
 
    // 画匹配图
    cv::Mat img_matches_knn;
    drawMatches( image1, keypoints1, image2, keypoints2, good_matches, img_matches_knn, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    cv::imshow("knn_matches", img_matches_knn);
    cv::waitKey(0);
 
 
 
    cv::Mat output;
    cv::drawKeypoints(image1, keypoints1, output);
    cv::imwrite("sift_image1_keypoints.jpg", output);
    cv::drawKeypoints(image2, keypoints2, output);
    cv::imwrite("sift_image2_keypoints.jpg", output);
 
 
    std::cout << "图1特征点检测耗时(ms)：" << tkpt << std::endl;
    std::cout << "图1特征描述符耗时(ms)：" << tdes << std::endl;
    std::cout << "BF特征匹配耗时(ms)：" << tmatch_bf << std::endl;
    std::cout << "KNN-NNDR特征匹配耗时(ms)：" << tmatch_knn << std::endl;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-16 11:20:31
 * @Parameters: 
 * @Return: 
 * @Description: get some face train samples from the movie. we will redefine
 * a function to detect face in picture or in movie. we have define the function and
 * have got many train face samples, then, we should define a function that can
 * get a txt file involved the face sample image path information and label.
 * 
 */
void getFaceSamplesFromMovie(const string moviePath, const string saveDirectory) 
{
    int count = 0;
    VideoCapture capture(moviePath);
    if (!capture.isOpened())
    {
        sys_error("cound not open camera...");
    }
    CascadeClassifier faceDetector;
    faceDetector.load(FACEDETECTMODEL2);
    Mat frame, image;
    vector<Rect> faces;
    while (capture.read(frame))
    {
        flip(frame, frame, 1);
        faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(100, 120), Size(300, 400));
        for (size_t i = 0; i < faces.size(); i++)
        {
            
            image = frame.clone();
            rectangle(image, faces[i], Scalar(0, 0, 255), 2, 8, 0);
            imshow("movie display", image);
            // this method will draw one face in one picture and wait to the input from terminal.
            // then draw the second face in the same picture. it will not
            // draw all faces in one frame. so you can select the face you want to
            // save it as samples.
            char c = waitKey(0); // wait forever.
            if (c == 32)
            {
                // save the detected face.
                Mat dest;
                string savePath = saveDirectory;
                resize(image(faces[i]), dest, Size(100, 100));
                savePath += "/face_";
                savePath += (++count);
                savePath += ".jpg";
                cv::imwrite(savePath, dest);
                cout << "save: " << savePath << endl;
            }
            else if (c == 27)
            {
                capture.release();
                return;
            }
            else if (c == 'n')
            {
                break;
            }
        }
    }
    capture.release();
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
 * @Date: 2023-02-16 14:40:24
 * @Parameters: directoryPath, involved all the train face samples. savePath
 * outgoing path.
 * @Return: 
 * @Description: this function will creat a txt file involved all the train face
 * image and corresponding labels.
 */
void getTrainDataFromeDir(const string directoryPath, vector<string> &imagePath) 
{
    // you should get all directory from the input directory.
    return;
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-14 19:27:56
 * @Parameters: string labelFile, you should pass the labelPath at least.
 * @Return: 
 * @Description: this file will test the application of face recognition used
 * EigenFace(statistic, PCA), FisherFace(linear discriminant analysis), LBPH(binary model
 * the advance is robustness for the light.)
 * and so on.
 * these method worked based on the statistic. it will work as follow step.
 * read the face train data -> calculate the average face -> convariance matrix
 * -> calculate the feature value and feature vector -> you will get a new feature
 * matrix based on the feature vector -> reduce the dimension used PCA -> get subspace model
 * -> test the distance -> judge the result.
 * 
 * then, we will test the face recognition efficient used eigenFace method.
 * of course, no matter to use any method, traditional method or neural network.
 * you must create the train sample. and the train sample must be recognized by
 * the classifier you have used it to detected the face. it means the efficient
 * of face classifier will influence your recognition efficient. so you should ensure your
 * sample is got by using the classifier.
 * 
 * we have created the train samples. then, we will train them used eigenFace method.
 * it will get the train model. then you can use the model to predict the face what 
 * you want to recignition.
 * 
 * then, we will need three function at least.
 * the first, you should define one function that can get all the train samples.
 * it means you can save the recgnized face in your disk if it is the train sample
 * what you want in a movie or a picture. of course, you should ensure the size 
 * of samples is similar to the predict sample as far as possibel.
 * 
 * second, you should create a directory that dedicated to saving all the samples 
 * name and the corresponding labels in one txt file. it will be used during trainng
 * and predicting.
 * 
 * third. you should define the function that can recognize all the face 
 * 
 * then we have defined all pretreatment working, we will start the face recognition function
 * based on them.
 */
void faceRecognitionUsedEigenFace(string labelFile, const string predictMoviePath) 
{
    ifstream file(labelFile.c_str(), ifstream::in);
    if (!file)
    {
        sys_error("could not load file correctly...");
    }
    string line, path, label;
    vector<Mat> images;
    vector<int> labels;

    char separator = ';';
    while (getline(file, line))
    {
        // set the path to path variabel, set label to label variabel.
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, label);
        if (!path.empty() && !label.empty())
        {
            images.push_back(imread(path, 0));
            // transform label from string to char used c_str().
            // then transformfrom char to int used atoi function.
            labels.push_back(atoi(label.c_str()));
        }
    }

    if (images.size() < 1 || labels.size() < 1)
    {
        sys_error("invaild image path...");
    }

    #if 0
    // test the model what has trained successful.
    Mat testImage = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    #endif

    // but you should also get the size of sample image. 
    // because you will use it during recognition. you should resize the detected face used the 
    // same size of sample.
    Mat sample = images[0];

    // load eigenFace, how to calculate the wasted time? we can calculate it if we define the
    // train function by ourselves.
    Ptr<face::BasicFaceRecognizer> model = face::EigenFaceRecognizer::create();
    model->train(images, labels);

    #if 0
    // test mode code 
    int predictedLabel = model->predict(testImage);
    printf("actual label: %d, predict label: %d", testLabel, predictedLabel);
    #endif

    // then, we will used the trained model to handle the movie.
    VideoCapture capture(predictMoviePath);
    CascadeClassifier faceDetector;
    faceDetector.load(FACEDETECTMODEL2);
    if (!capture.isOpened())
    {
        sys_error("could not open the movie...");
    }
    Mat frame, dest;
    namedWindow("face recognition", WINDOW_AUTOSIZE);
    vector<Rect> faces;
    while (capture.read(frame))
    {
        flip(frame, frame, 1);
        faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(80, 100), Size(380, 400));
        for (size_t i = 0; i < faces.size(); i++)
        {
            Mat roi = frame(faces[i]);
            cvtColor(roi, dest, COLOR_RGB2GRAY);
            resize(dest, sample, sample.size());
            int predictLabel = model->predict(sample);

            rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
            string text;
            if (predictLabel == 1)
            {
                text = "gtl";
            }
            else if (predictLabel == 2)
            {
                text = "gyy";
            }
            else
            {
                text = "somebody";
            }
            putText(frame, text, faces[i].tl(), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            imshow("face recognition", frame);
            if (waitKey(50) == 27)
            {
                break;
            }
        }
    }
}



/**
 * @Author: weiyutao
 * @Date: 2023-02-15 19:06:30
 * @Parameters: 
 * @Return: 
 * @Description: we will use the professional image recognition module dlib.
 * because of the namespace of dlib is conficted with vector in std. so you should
 * drop one. you can drop to use the namespace dlib.
 */
void faceDetectUsedDlib(Mat &inputImage, Mat &outputImage, int mode) 
{
    outputImage = inputImage.clone();
    // Dlib HoG face detected algorithm what is the most efficient algorithm in cpu. it can detect
    // slight positive face. but the HoG model can not detect the small size face. you can train yourself
    // small size face classifier if you want enhance its efficient. of course, we can
    // use anthor algorithm that dlib has provided. MMOD dlib_dnn model. it is more efficient.
    // and support to run in the GPU.
    if (mode == DLIB::HOG)
    {
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::cv_image<dlib::bgr_pixel> image(outputImage);
        std::vector<dlib::rectangle> faces = detector(image);
        for (unsigned int i = 0; i < faces.size(); i++)
        {
            cv::rectangle(outputImage, cv::Rect(faces[i].left(), faces[i].top(), \
            faces[i].width(), faces[i].width()), cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }
    else if(mode == DLIB::MMOD)
    {
        // string mmodModel = "../resources/model/mmod_human_face_detector.dat";
        // dlib::null_trainer_type<> mmodFaceDetector;
        // dlib::deserialize(mmodModel) >> ;
    }
}