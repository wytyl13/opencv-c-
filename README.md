# opencv/c++
how to install opencv in windows.
first, you should download compiler, just like mingw that support posix version.
    because the opencv need to the support that multi threads. download the newest
    version and add the bin directory that involved all the binary executable command into the system environment variabel. in order to you can call them in any directory in your computer in your terminal.
second, you should download cmake, what you need to generate the makefiles used it.
    download the newest version. it can generate makefile file based on the cmakelists.txt file. it can also install the compiled sources in your disk.
third, you should download the sources about opencv and opencv-contrib. the former
    is the sources about opencv and the last is the expansion pack, because the function in it is unstable, so it is placed on the expansion sources. notice, 
    the version of these two sources must be consistant, or you will get the error
    during installing the opencv. because some head files is not agree with the implementation files.
fourth, you should mkdir one directory what you installing location for opencv.
    the include files and lib will all be in this location. no matter you used 
    what name to create this directory, you will find the opencv dependency in the install directory in this directory.
fifth, all preparatory work are ok, then you should compiler opencv and     opencv-contrib.
    just like yourself program, you should compile first, then execute the binary file.
    install the source is the similar, compiled it used cmake -G "MinGW Makefiles", notice you should ensure the 
    your current directory has the cmakeLists.txt file, or you will add the extra param after this cmake command, because this cmake command will generate makefile based on the file cmakeLists.txt, and cmakeLists.txt involved some setting about dependency of program internal and external. makefile is the compile command setting. just like cmake -G "MinGW Makefiles" .., it means you will generate the makefile file based on the
    cmakeLists.txt file where cmake command will find it at the next higher level
    directory. cmake command is equal to generate option in cmake-gui, and ./configure is equal to configure option in cmkae-gui. then, how to generate the cmakeLists.txt file? you need not to consider it when you want to install
    the third-party libraries, because its program will involved it.

    then, this command configure will test all necessary files and environment what the opencv program needed in your computer, if this test has not passed, you will failed to compile this program. of course, you should add the opencv-contrib path and select the build_opencv_world option to generate one library when you compiled. of course, you can test to use cmake-gui when you want to add these options.

    then, image we have passed ./configure and cmake -G "MinGw Makefile" .., these two command, you should input make, it will compile your process based on the makefile what cmake -G "MinGw Makefile" .. command generated based on cmakeLists.txt file. util here you have got the binary file that compiled based on makefile file. you should input make install command if you want to 
    install this process. you should make run this binary file if you do not want to install the program.
sixth, the command above is suitable for the third-party library and youself process.


update the linearScale and bianryScale to interpolate the image
update some defination about digital image
update the image transform used affine matrix, involved manual define and use the offficial function
add some contents about standard transform for image and Eigen model
add the content of gray layered based on the points, gray value and bit planes
add the histogram equalization transform
add the histogram specification transform
add the local histogram transformation
add the histogram transformation with statistic. it is similar to histogram transformation with conditional.
add the spatial filter, it is equal to the kernal, convolution kernels, it is not
equal to the local histogram transformation. the former involved scan and kernel, these two local matrix, you can transform by different calculation method, just like dot, you can also transform by setting the different kernel matrix. the last
method just involved scan matrix, then you can just transform the original gray value based on the feature that you have scanned, the feature just like histogram.
you can do histogram equalization and histogram specification and so on.
add some super application about feature extraction and face recognition, the feature extraction involved feature detected, chain code and the super application boundary detected about it. the face recognition involved face detected and face compare, we have used the eigenFace method what is a statistic recognition method, it has defined by opencv official. the next plan, we will learn some other face detected and recognition method, just like some method has defined by dlib. and we will learn some feature detected algorithm, these content should be learned in digital image process book.
update the spatial filter kernel, and deep learning how to define the filter kernel.
and analyze some words, just like filter, kerenl, high-pass, operator, low-pass and so on.
add the derivative of gray value in image, add four laplacian operators and implement the image of sharpening used them.





#some problem what have not handled in this process you should focus on it.
1 how to define the operators based on the existing requirements.
2 


add the function sharpenImageUsedPassivationTemplate, and test the influence that the degree of sharpening by the degree of fuzzying and k value
