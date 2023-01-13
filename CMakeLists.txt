cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

#echo the version about opencv
MESSAGE("OpenCV version : ${OpenCV_VERSION}")
MESSAGE("Eigen3 version : ${Eigen3_VERSION}")

#set the project_name is opencv for the variable PROJECT_NAME
project(opencv)
#file(GLOB_RECURSE SRC_DIR_LIST "src/*.cpp" "src/*.c")扫描目录下的所有文件存入变量

#add the head file path
include_directories(./include)#指定头文件，相当于g++中的-I

#add the param
add_compile_options(-Wall -fexec-charset=GBK)

#store all the cpp file need to be compiled into the variable SRC_DIR_VAR.
aux_source_directory(./src SRC_DIR_VAR)#扫描src目录下的所有文件并保存到变量中
# aux_source_directory(./include SRC_DIR_VAR)#扫描src目录下的所有文件并保存到变量中
#target_sources(${target} PRIVATE ${SRC_DIR_VAR})

#the rule to generate opencv.exe
add_executable(opencv ${SRC_DIR_VAR})#使用传入变量的方式加入要编译的文件

#define the OpenCV_DIR variable ie equal to D:/development_app2/opencv/build/install
#it is the path of opencv library
# notice. this variable is the include path for opencv and Eigen3
# the absolutely include path of Eigen3 is D:/development_app2/Eigen3/Eigen
# all the include file is stored in this path. we have defined the variable
# EIGEN3_INCLUDE_DIR used D:/development_app2/Eigen3, and include_directories it
# so then we should include the head file Eigen/Dense, just like
#include<Eigen/Dense>, include it in the program, it is samed to include
# D:/development_app2/Eigen3/Eigen/Dense, because we have include_directories
# D:/development_app2/Eigen3, so we just need to fill gas the back of path.
# notice. Eigen is not compile the installation but the unzip the installation.
set(OpenCV_DIR D:/development_app2/opencv/build/install)
set(EIGEN3_INCLUDE_DIR D:/development_app2/Eigen3)
#scan the package
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
#if is exist
if((OpenCV_FOUND) AND (Eigen3_FOUND))
    message(STATUS "found OpenCv: " ${OpenCv_INCLUDE_DIR})
    message(STATUS "found Eigen3: " ${EIGEN3_INCLUDE_DIR})
    #add the include path
    include_directories(${OpenCv_INCLUDE_DIR})
    include_directories(${EIGEN3_INCLUDE_DIR})
    #add the lib
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    # because Eigen is not involved link, so we need not to include it's link libraries.
    # target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBS})
else()
    message(FATAL_ERROR "not found!")
endif()

#why opencv can do this? because the path structure is fixed.
#notice, you must add the add_executable(opencv ${SRC_DIR_VAR}) what is the rule to do, before the
#你应该把编译规则定义在定义链接库地址的前面，因为动态链接库在编译时候才可以知道，而前面的头文件位置在编译前就知道。
#这也就解释了为什么之前学习的，动态链接库的定位晚于头文件和静态链接库的定位寻址。
# if(OpenCV_FOUND)
#     #add the include path
#     include_directories(${OpenCV_INCLUDE_DIRS})
#     #add the lib
#     target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# endif()