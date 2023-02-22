cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

#echo the version about opencv
MESSAGE("OpenCV version : ${OpenCV_VERSION}")
MESSAGE("Eigen3 version : ${Eigen3_VERSION}")

#set the project_name is opencv for the variable PROJECT_NAME
project(opencv)
#file(GLOB_RECURSE SRC_DIR_LIST "src/*.cpp" "src/*.c")扫描目录下的所有文件存入变量

#add the head file path
# notice, you just need to add the include path for your program. 
# the cmakelists.txt file will generat the makefile that invovled your program's include
# and other three party libraries' include path. you should command make all in your terminal.
# and the generated makefile file is located at build directory.
# you should make all at there. if you use g++ in your terminal, you should add
# -I to specified the include path and -l specified the dynamic library name and 
# -L to specified the dinamic library directory.
include_directories(./include)#指定头文件，相当于g++中的-I

#add the param
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)

#store all the cpp file need to be compiled into the variable SRC_DIR_VAR.
aux_source_directory(./src SRC_DIR_VAR)#扫描src目录下的所有文件并保存到变量中
# aux_source_directory(./include SRC_DIR_VAR)#扫描src目录下的所有文件并保存到变量中
#target_sources(${target} PRIVATE ${SRC_DIR_VAR})

#the rule to generate opencv.exe
add_executable(opencv ${SRC_DIR_VAR})#使用传入变量的方式加入要编译的文件

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
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
set(OpenCV_DIR C:/Users/weiyutao/opt/opencv/build/install)
set(EIGEN3_INCLUDE_DIR C:/Users/weiyutao/opt/eigen-3.4.0)
set(Dlib_INCLUDE_DIR "C:/Users/weiyutao/opt/dlib-19.24/dlib-19.24/install/Program Files (x86)/dlib_project/include")
set(Dlib_LIBS "C:/Users/weiyutao/opt/dlib-19.24/dlib-19.24/install/Program Files (x86)/dlib_project/lib/libdlib.a")
#the addtional library for opencv. you can include its head file in your program
set(OpenCV_Contrib "C:/Users/weiyutao/opt/opencv/source/opencv_contrib-4.7.0/modules")
# set the static library path, you can set the lib path and added it used
# target_link_libraries(${PROJECT_NAME} ${Dlib_LIBS} X11 pthread), you should 
# use find_package(dlib REQUIRED) and target_link_libraries(${PROJECT_NAME} ${dlib_LIBRARIES})
# to set it. this is a dlib static library, it is very slowly when you compiler your
# program. you can set the libarary used set command. you can also use find_package.
# just like the code as follow.

# find_package(dlib REQUIRED)
# include_directories(${dlib_INCLUDE_DIR})
# target_link_libraries(${PROJECT_NAME} dlib::dlib)

# target_link_libraries(${PROJECT_NAME} ${Dlib_LIBS})

#scan the package
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)



#if is exist
# AND (Eigen3_FOUND) AND (Dlib_FOUND)
if((OpenCV_FOUND))
    message(STATUS "found OpenCV: " ${OpenCV_INCLUDE_DIR})
    message(STATUS "found OpenCV: " ${OpenCV_LIBS})
    message(STATUS "found Eigen3: " ${EIGEN3_INCLUDE_DIR})
    message(STATUS "found Dlib: " ${Dlib_INCLUDE_DIR})
    #add the include path
    include_directories(${OpenCV_INCLUDE_DIR})
    include_directories(${EIGEN3_INCLUDE_DIR})
    include_directories(${Dlib_INCLUDE_DIR})
    include_directories(${OpenCV_Contrib})
    # cpp call python script will use the python head file, what is lacated at the 
    # directory as follow.
    include_directories("C:\\Users\\weiyutao\\opt\\python\\include")
    #add the lib
    # target_link_libraries(${PROJECT_NAME} ${Dlib_LIBS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    target_link_libraries(${PROJECT_NAME} ${Dlib_LIBS})
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