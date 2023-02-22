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

#if ISOPENDLIB
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
        face_cascade.load(OPENCVHAARFACEDETECT);
    }
    if (eye_cascade.empty())
    {
        eye_cascade.load(OPENCVHAAREYEDETECT);
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
    int width = 0;
    int height = 0;
    int total_frame;
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
    faceDetector.load(OPENCVHAARFACEDETECT_EXTRA);
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

    #if 0
    // of course, you can also use anthor two face recognition algorithm.
    // FisherFace algritnm is the best face recognition method.
    // LBPH is the best face detected algorithm.
    Ptr<face::BasicFaceRecognizer> model = face::FisherFaceRecognizer::create();
    Ptr<face::BasicFaceRecognizer> model = face::LBPHFaceRecognizer::create();
    #endif
    model->train(images, labels);

    #if 0
    // test mode code 
    int predictedLabel = model->predict(testImage);
    printf("actual label: %d, predict label: %d", testLabel, predictedLabel);
    #endif

    // then, we will used the trained model to handle the movie.
    VideoCapture capture(predictMoviePath);
    CascadeClassifier faceDetector;
    faceDetector.load(OPENCVHAARFACEDETECT_EXTRA);
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
 * drop one. you can drop to use the namespace dlib. you need not to load the face classifier, 
 * because dlib has contained them.
 */
void faceDetectUsedDlib(Mat &inputImage, Mat &outputImage, int mode) 
{
    outputImage = inputImage.clone();
    // Dlib HoG face detected algorithm what is the most efficient algorithm in cpu. it can detect
    // slight positive face. but the HoG model can not detect the small size face. you can train yourself
    // small size face classifier if you want enhance its efficient. of course, we can
    // use another algorithm that dlib has provided. MMOD dlib_dnn model. it is more efficient.
    // and support to run in the GPU. the default method is HOG algorithm.
    // the mode SHAPE68 can detect 68 features in face.
    // you should detect all the faces from the picture used detector what you have created. 
    // then detected the feature from each face used shapePredict you have defined.
    // of course, you should defined dilib type container to accept these return value.
    // and you should loaded the trained model dilb website provided.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::cv_image<dlib::bgr_pixel> image(outputImage);
    std::vector<dlib::rectangle> faces = detector(image);
    if (mode == DLIB::HOG)
    {
        for (unsigned int i = 0; i < faces.size(); i++)
        {
            cv::rectangle(outputImage, cv::Rect(faces[i].left(), faces[i].top(), \
            faces[i].width(), faces[i].width()), cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }
    else if(mode == DLIB::MMOD)
    {
        // you can use mmod what a trained successful neural network model.
        // the former method is HOG what is not the neural network, it is used
        // characteristics of cascade classifier. the method to define them are different.
        dlib::shape_predictor sp;
        dlib::deserialize(DLIBMMODMODELFACEDETECT) >> sp;
        return;
    }
    else if (mode == DLIB::SHAPE68)
    {
        // you should add the extra detected model if you want to detect the 68 features in face.
        // of course, you should detected the face first. and then detected 68 features based on
        // the detected face.
        dlib::shape_predictor shapePredict;
        dlib::deserialize(DLIBFACEFEATUREDETECT) >> shapePredict;
        vector<dlib::full_object_detection> shapes;
        for (size_t i = 0; i < faces.size(); ++i)
        {
            shapes.push_back(shapePredict(image, faces[i]));
        }
        for (size_t j = 0; j < faces.size(); ++j)
        {
            if (!shapes.empty())
            {
                for (int i = 0; i < 68; i++)
                {
                    string text = "" + i;
                    cv::Point center = cv::Point(shapes[j].part(i).x(), shapes[j].part(i).y());
                    cv::circle(outputImage, center, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
                    /* putText(outputImage, text, center, FONT_HERSHEY_SCRIPT_SIMPLEX,\
                        0.5, cv::Scalar(0, 0, 255), 0.001, 8); */
                }
            }
            imshow("dlib feature detect", outputImage);
            #if 0
            // the link error happend when added these follow code. but we have linked the library of dlib.
            // so we will try to use another method.
            dlib::image_window window(image);
            window.add_overlay(dlib::render_face_detections(shapes));
            window.wait_until_closed();
            #endif
        }
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-17 10:42:49
 * @Parameters: 
 * @Return: 
 * @Description: the resnet mdoel of face recognition in dlib used resnet neural network. the concept of
 * resnet is to improve the feature detected rate than other traditional digital image process algorithm
 * or the original neural network. it means you can get more feature vector about one face and compare
 * it with the dest image, you can get more correct recognition result. then, we will deep describe the face
 * recognition concept.
 * 
 * step1, detected the face. you can reduce your working and improve your correct rate if you can ignore
 * the other interference factors.
 * 
 * step2, accurate to face, detected all features of face as far as possible. more features more accuracy.
 * you can use the professional face feature detected, just like shape_predictor_68_face_landmarks.dat, or
 * dlib_face_recognition_resnet_model_v1.dat. of course, you can also use the feature detected method in opencv.
 * of course, it is generally suitable for the feature detected for all object in one picture, it is not dedicated
 * to detecting the face feature.
 * 
 * step3, you have got the face features, then, you should compare them based on the error of feature vectors
 * between the sample and dest. so we can conclude that the difference between the original face recognition and
 * the neural network. the traditional used some dimension reduction method, just like PCA. so it can enhance 
 * the efficient but low accuracy. but resnet model remain the important information of face. so you can use it
 * to get large accuracy. it means the difference between these two method is feature vectors.
 * 
 * step4, compare the feature vectors, you can image that as euclidean distance between two feature vectors.
 * 
 * we have implemented these steps above in the former function used opencv. we used haar classifier to 
 * detected the face, and calculated the feature vectors based on the convariance matrix. then use PCA method
 * to recognize. then, we will use the resnet method in dlib to implement the face recognition function.
 * 
 * in order to use the trianed successful model resnet in dlib, you should define a data type that
 * suitable for it. we have defined it in genearl head file based on the fixed standard about resnet
 * in dlib.
 */
void faceImageRecognitionUsedDlib(const string dirPath, const string targetImagePath) 
{
    Mat image;
    // this a column vector, float dimension is (0, 1).
    vector<dlib::matrix<float, 0, 1>> featureVectors;  // store all the feature vectors of all detected faces.
    float vector_error[30];
    int count = 0;
    int invalidCount = 0;

    #if 1
    // scan all the picture, and stored them, and record the numbers.
    std::vector<cv::String> fileNames, imagePaths;
    getImageFileFromDir(dirPath, fileNames, imagePaths, count);
    // the default sort megthod is dictionary order, you should define the compare rule function
    // if you want to implement the complex sort rule. pass the compare function into the third function.
    // sort(fileNames.begin(), fileNames.end(), compareVectorString);
    // printVector(fileNames);
    #endif

    #if 1
    // load all model, involved face detected, feature detected and resnet model.
    // use HOG method what is a cascade of face classifier in dlib
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shapePredict;
    dlib::deserialize(DLIBFACEFEATUREDETECT) >> shapePredict;
    anet_type net;
    dlib::deserialize(DLIBRESNETMODEL) >> net;
    // it is equal to Mat image, it is the image type for dlib.
    #endif

    #if 1
    // this step we can name it as detected face and calculated feature vectors.
    string imagePath; // stored the current index k imagePath.
    std::vector<dlib::rectangle> dest; // stored the detected faces of the current index k image.
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces; // stored all the face image
    dlib::full_object_detection shape; // stored the 68 feature data of the current face.
    dlib::matrix<dlib::rgb_pixel> image_dlib, faceImage; // store the current overall image and the face image what got from the current overall image.
    std::vector<dlib::matrix<float, 0, 1>> faceDescriptors; // store all the feature vectors. from index 0 to current index k.
    // read all picture what we have stored them in imagePath.
    for (int k = 0; k < count; k++)
    {
        imagePath = imagePaths[k];
        dlib::load_image(image_dlib, imagePath); // read image and stored used image_dlib.
        std::vector<dlib::rectangle> dest = detector(image_dlib); // detected face used HOG in dlib
        if (dest.size() < 1)
        {
            cout << "handling " << imagePath << ", there is not face, ignored..." << endl;
            invalidCount++;
        }
        else if (dest.size() > 1)
        {
            cout << "handling " << imagePath << ", detected many faces, ignored..." << endl;
            invalidCount++;
        }
        else
        {
            // size == 1, you can go on working. define the variable to store each face.
            // you should get the face from one image, so you should store it as one image.
            // of course, you can also use auto what is the new feature in cpp11.
            // because you can ensure just has one face, so you just need to define one shape.
            // need not to define a vector to store it.
            shape = shapePredict(image_dlib, dest[0]); // get the feature of face.
            // then, you should get the face region as one image from original image.
            dlib::extract_image_chip(image_dlib, dlib::get_face_chip_details(shape, 150, 0.25), faceImage);
            faces.push_back(move(faceImage));
            faceDescriptors = net(faces);
            featureVectors.push_back(faceDescriptors[k - invalidCount]);
            cout << "the vector of picture " << imagePaths[k] << endl;
        }
    }
    #endif

    #if 1
    // this step we can name it as recognition. but you should get the feature vector of the target image.
    // notice, the size of face could not be small if you detected faces used dlib library.
    // because the dlib library does not support the detected about small size face.
    Mat targetImageMat = cv::imread(targetImagePath);
    if (targetImageMat.empty())
    {
        sys_error("load Mat error, please chech your code or imagePath...");
    }
    dlib::cv_image<dlib::bgr_pixel> targetImageDlib(targetImageMat);
    dlib::matrix<dlib::rgb_pixel> targetFaceImage;
    std::vector<dlib::matrix<dlib::rgb_pixel>> targetFacesImage;
    // detected all faces in this target image, and intercepted then from target image and 
    // stored them used faces_test variable.
    std::vector<dlib::rectangle> faces_test = detector(targetImageDlib);
    for (auto face_test : faces_test)
    {
        auto shape_test = shapePredict(targetImageDlib, face_test);
        dlib::extract_image_chip(targetImageDlib, dlib::get_face_chip_details(shape_test, 150, 0.25), targetFaceImage);
        targetFacesImage.push_back(move(targetFaceImage));
    }
    // resnet the faces_test variable.
    // notice, the input face image size must be equal to the model anet_type we have defined in head file.
    // or you will get the error: Failing expression was i->nr()==NR && i->nc()==NC.
    // All input images must have 150 rows and 150 columns, but we got one with 159 rows and 159 columns.
    std::vector<dlib::matrix<float, 0, 1>> targetFaceDescriptors = net(targetFacesImage);
    cout << "the numbers of face in target image is: "<< targetFaceDescriptors.size() << endl;
    #endif

    #if 1
    // recognition. compare these two feature vector.
    cv::Point origin;
    int width = 0;
    std::string text;
    for (size_t i = 0; i < targetFaceDescriptors.size(); i++)
    {
        origin.x = faces_test[i].left();
        origin.y = faces_test[i].top();
        width = faces_test[i].width();
        text = "anybody";
        for (size_t j = 0; j < featureVectors.size(); j++)
        {
            vector_error[j] = (double)dlib::length(targetFaceDescriptors[i] - featureVectors[j]);
            if (vector_error[j] < 0.4)
            {
                text = fileNames[j];
                cout << "find:" << fileNames[j] << "," << text << endl;
            }
        }
        cv::putText(targetImageMat, text, origin, FONT_HERSHEY_SIMPLEX,\
                0.5, Scalar(255, 0, 0), 2, 8, 0);
        cv::rectangle(targetImageMat, Rect(origin.x, origin.y, width, width), cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    #endif

    imshow("result image", targetImageMat);
}



/**
 * @Author: weiyutao
 * @Date: 2023-02-20 10:19:38
 * @Parameters: 
 * @Return: 
 * @Description: then, this is the function that recognized faces. it involved face detected, face recogninzed.
 * face detected involved feature detected, we used HOG method in dlib here. of course, you can also use
 * haar method in opencv, you should read the xml file in opencv, but you can use the HOG directly and need not
 * to load the corresponding xml file. of course, these two method can detected faces in one image. because the opecv
 * and dlib has encapsulated the feature deteceted algorithm, and trained the classifier of face.
 * these two method can dedicated to using at face detected. because they are all classifier for faces.
 * 
 * then, we used shapePredict in dlib detected the face features in one image. it can return the more accurate
 * face features. notice, the difference between shapePredict and classifier( just like HOG in dlib and haar in opencv).
 * shapePredict is also a classifier for face, but it can also give all the 68 features in face. we can use it to get
 * the feature vectors for one face used resnet. it means we can get the face features and use neural network to
 * get the more accurated feature vectors, it is different from the PCA method to calculate the feature vectors.
 * PCA used the dimension reduction method. the neural network has more accurated result.
 * 
 * then, we compared the feature vector of src image and target image. and define the threshold value to
 * judge the face recognition result.
 * 
 * so, we have considered the feature detected method. we should learn them at last.
 * 
 * we have learned how to classificate one image, we can train the classifier based on these features.
 * just like you can train a face classifier to detecet faces in one image, just like HOG in dlib, haar in opencv.
 * they are all in xml file form to show for us. we can use dlib::get_frontal_face_detector method in dlib to get the HOG detector.
 * it is dlib::frontal_face_detector class. we can also use faceDetector.load(OPENCVHAARFACEDETECT_EXTRA);
 * to load the haar face deteceted classifier, it is the CasecadeClassifier class in opencv. so we will
 * learn how to train the classifier at last. the classifier in opencv is very low efficient, it is 
 * efficient in opencv. just like you can label the train samples used imglab tool. and then train them used
 * train_... in dlib.
 * 
 * we have learned how to recognize one image, we will compare two feature vector. the feature vector we can use
 * PCA, it is generally used as eigenFace class in opencv, the concept of eigenFace is PCA. just like one label, 
 * you can give some samples, you can use PCA to these samples. calculated the convariance matrix about these samples.
 * then calculated the feature vectors based on the convariance matrix, then compare the feature vector.
 * and judge the label based on the compared result error. you can find the more accurated result if you passed
 * more samples, but the accurated also has the ceiling. because PCA method droped some information of the face.
 * but the neural network method in dlib is different from the traditional recognition method. we will use
 * shapePredict method to get 68 features in face. and passed them into the neural network and get the more ditailed
 * feature value and feature vector. this method do not use the dimension reduction, so it will has more auurated.
 * 
 * util here, you can find the difference between traditional face recognition method and neural network is
 * the different method about the feature vectors. you can has the detailed steps as follow.
 * 
 * first, we will start with the simple face detected.
 * 1 load the detector used haar xml files in opencv, or used HOF xml file in dlib. they are all the classifier.
 * 2 detector(image) this method can return all the faces in one image. you should use cv::Rect or dlib::rectangle to
 *  accept each detected face.
 * 
 * second, we will consider the face recognition.
 *  calculated the feature vectors of each face. of course, you should pass the face image that got at the former handle.
 *  you can not pass the overall image.
 *  two method, PCA and neural network.
 *  PCA, you should give some face samples and the corresponding labels. train them used eigenFace in opencv.
 *  then predict the label of traget image used predict method.
 *  neural network, just like resnet, you should load the trained successful xml file. you can get the feature vectors
 *  used the net class. but you should pass the 68 shape features into the net, so you should load the 68 feature
 *  xml file used full_object_detection class. the you will get the vector features used net. then, this method
 *  need not give some samples. you can give one sample. then, you should get the feature vector of target image used
 *  the same net, at last, you should compare the error between the feature vector of sample and target, and judge
 *  the predict label used the error.
 * 
 * third, we will consider the detected about the other object.
 *      this case just involved the object detected, do not involved recognition. so you should just train
 *      the classifier or the model. of course, you can also use neural network. but we will just consider
 *      how to train the classifier of any object.
 *      method1, you can used the classfier method in opencv. but it is low efficient. and it is dropped by
 *      the high version opencv. so we will consider it used dlib.
 * then, we will consider the detected about any object in dlib. it means we will make the classifer by ourselves,
 * because the face detecetd classifer is generally trained successful. you should
 * train yourself classfier of specific object if you want to detected the other object.
 * we will step out of the application of the face detected and recognition, and return to the 
 * digital image processing. we will learn it from the spatial filter what is located at 
 * the chapter3 in digital image processing bool.
 */
void faceMovieRecognitionUsedDlib(const string dirPath, const string targetMoivePath)
{
    vector<dlib::matrix<float, 0, 1>> featureVectors;
    float vector_error[30];Mat image;int count = 0;int invalidCount = 0;

    #if 1
    std::vector<cv::String> fileNames, imagePaths;
    getImageFileFromDir(dirPath, fileNames, imagePaths, count);
    #endif

    #if 1
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shapePredict;
    dlib::deserialize(DLIBFACEFEATUREDETECT) >> shapePredict;
    anet_type net;
    dlib::deserialize(DLIBRESNETMODEL) >> net;
    #endif

    #if 1
    string imagePath; 
    std::vector<dlib::rectangle> dest;
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    dlib::full_object_detection shape;
    dlib::matrix<dlib::rgb_pixel> image_dlib, faceImage;
    std::vector<dlib::matrix<float, 0, 1>> faceDescriptors;
    for (int k = 0; k < count; k++)
    {
        imagePath = imagePaths[k];
        dlib::load_image(image_dlib, imagePath);
        std::vector<dlib::rectangle> dest = detector(image_dlib);
        if (dest.size() < 1)
        {
            cout << "handling " << imagePath << ", there is not face, ignored..." << endl;
            invalidCount++;
        }
        else if (dest.size() > 1)
        {
            cout << "handling " << imagePath << ", detected many faces, ignored..." << endl;
            invalidCount++;
        }
        else
        {
            shape = shapePredict(image_dlib, dest[0]);
            dlib::extract_image_chip(image_dlib, dlib::get_face_chip_details(shape, 150, 0.25), faceImage);
            faces.push_back(move(faceImage));
            faceDescriptors = net(faces);
            featureVectors.push_back(faceDescriptors[k - invalidCount]);
            cout << "the vector of picture " << imagePaths[k] << endl;
        }
    }
    #endif

    #if 1
    cv::VideoCapture capture(targetMoivePath);
    if (!capture.isOpened())
    {
        sys_error("unable to connect to camera...");
    }
    dlib::array2d<dlib::rgb_pixel> targetImageDlib;
    dlib::matrix<dlib::rgb_pixel> targetFaceImage;
    std::vector<dlib::rectangle> faces_test;
    std::vector<dlib::matrix<float, 0, 1>> targetFaceDescriptors;
    int width = 0;std::string text;cv::Mat targetImageMat;

    while (waitKey(1) != 27)
    {
        if (!capture.read(targetImageMat))
        {
            capture.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        dlib::assign_image(targetImageDlib, dlib::cv_image<dlib::rgb_pixel>(targetImageMat));
        faces_test = detector(targetImageDlib);
        std::vector<dlib::matrix<dlib::rgb_pixel>> targetFacesImage;
        for (auto face_test : faces_test)
        {
            auto shape_test = shapePredict(targetImageDlib, face_test);
            dlib::extract_image_chip(targetImageDlib, dlib::get_face_chip_details(shape_test, 150, 0.25), targetFaceImage);
            targetFacesImage.push_back(move(targetFaceImage));
        }
        targetFaceDescriptors = net(targetFacesImage);
        cout << "the numbers of face in target image is: "<< targetFaceDescriptors.size() << endl;
        for (size_t i = 0; i < targetFaceDescriptors.size(); ++i)
        {
            width = faces_test[i].width();
            text = "anybody";
            for (size_t j = 0; j < featureVectors.size(); j++)
            {
                vector_error[j] = (double)dlib::length(targetFaceDescriptors[i] - featureVectors[j]);
                if (vector_error[j] < 0.4)
                {
                    text = fileNames[j];
                    cout << "find:" << fileNames[j] << "," << text << endl;
                }
            }
            cv::putText(targetImageMat, text, cv::Point(faces_test[i].left(), faces_test[i].top()), \
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2, 8, 0);
            cv::rectangle(targetImageMat, Rect(faces_test[i].left(), faces_test[i].top(), width, width),\
                cv::Scalar(0, 0, 255), 1, 8, 0);
        }
        imshow("result movie", targetImageMat);
    }
    #endif
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-20 11:20:29
 * @Parameters: 
 * @Return: 
 * @Description: you should train the model first.
 * but you should define the any shape object, we can not simple used face detected method to get 
 * the face in one movie, because we have not got the classifier of any shape file. so we should 
 * mark the shape used imglab tool. you can mark the shape and mark the features in the range of the shape.
 * the more feature you marked the more auurated detected rate.
 * 
 * then, you should train these samples you have got used imglab used train_object_detector what is 
 * the cpp file in example directory in dlib. you should compiler it first. you can compiler the train_object_detector.cpp
 * file used the cmakelists.txt file. the step is cmake -G "MinGW Makefiles", if you want to compiler it into
 * build directory, you can mkdir build first at the example directory. and cd build, cmake -G "MinGW Makefiles" ..
 * this command will compiler these cpp file in the build directory. and generate the executable file
 * train_object_detector.exe, then, you can use this exe file to train your samples created used imglab.exe
 * you can compiler all file and you can also just compiler the train_object_detector file.
 * make train_object_detector or make all or make. then, you will get the train_object_detector.exe file in
 * the build directory. the absolutely path is C:\Users\weiyutao\opt\dlib-19.24\dlib-19.24\examples\build\train_object_detector.exe
 * 
 * at last, you will get the classifier xml file, you can use it to create the classifier in your dlib
 * program.
 * 
 * 
 * the step is as follow:
 * step1, find the samples image. and stored them into one directory. then use imglab command.
 *      imglab -c shape.xml yourDir, this command will generate the original xml file involved all the image in yourDir.
 * step2, imglab ./shape.xml. edit the shape.xml file. and rectangle your interested region.
 * step3, imglab ./shape.xml --parts "1 2 3 4 5 ...". you can define some key feature point to imporve
 *      your recognition accurately. then, you should edit the xml file again and set the key feature points
 *      in the rectangle that you have draw in the image.
 * step4, when you have successful modified your xml file, then you should train it used train_object_detector.exe
 *      what is a executable file you have compiled used g++. then this command will generate one svm file.
 *      it is the classifier file based on the svm model.
 * step5 at last. you can use train_object_detector yourTargetImage.jpg to predict the shape you interested
 *      in the target image. if the shape you interested is existed in the image, this command will rectangle
 *      it in the image used red box and imshow the image in the current terminal.
 * 
 * then, we will close all dlib code in this program. because the static library will influence 
 * the speed of compilering.
 * and these parts about feature detected and face application is blong
 * to the image cutting and feature extraction and image pattern classification.
 * these three chapter in digital image processing book. so we will deeping learn
 * them at last. we will create these three part program file first, and then return
 * to the chapter3 of digital image processing what is spacial filter.
 */
void anyObjectDetectedUsedDlib(Mat &inputImage, Mat &outputImage) 
{   
    // you can resize the inputImage, in order to improve the accurate of recognizing.
    outputImage = inputImage.clone();
    Mat resizeImageMat = resizeImage(inputImage, 0.5);
    dlib::array2d<dlib::rgb_pixel> imageDlib;
    dlib::assign_image(imageDlib, dlib::cv_image<dlib::bgr_pixel>(resizeImageMat));
    typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
    ifstream fin_walk("C:/Users/weiyutao/development_code_2023-01-28/vscode/resources/trainSample/trainDlib/walk/object_detector.svm");
    // image_scanner_type is located at #include <dlib/image_processing/object_detector.h>
    dlib::object_detector<image_scanner_type> detector_walk;
    // deserilize the svm file what we have trained successful.
    dlib::deserialize(detector_walk, fin_walk);
    std::vector<dlib::rectangle> detect_result = detector_walk(imageDlib);
    if (detect_result.size() > 0)
    {
        cv::rectangle(outputImage, Point(detect_result[0].left(), detect_result[0].top()), Point(detect_result[0].width(),\
             detect_result[0].height()), Scalar(0, 0, 255), 1, 8);
    }
}
#endif