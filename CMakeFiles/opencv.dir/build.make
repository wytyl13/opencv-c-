# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\cmake-3.23.3-windows-x86_64\cmake-3.23.3-windows-x86_64\bin\cmake.exe

# The command to remove a file.
RM = D:\cmake-3.23.3-windows-x86_64\cmake-3.23.3-windows-x86_64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\development_code_2022-9-10\vscode\opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\development_code_2022-9-10\vscode\opencv

# Include any dependencies generated for this target.
include CMakeFiles/opencv.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencv.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv.dir/flags.make

CMakeFiles/opencv.dir/src/bitOperation.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/bitOperation.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/bitOperation.cpp.obj: src/bitOperation.cpp
CMakeFiles/opencv.dir/src/bitOperation.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv.dir/src/bitOperation.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/bitOperation.cpp.obj -MF CMakeFiles\opencv.dir\src\bitOperation.cpp.obj.d -o CMakeFiles\opencv.dir\src\bitOperation.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\bitOperation.cpp

CMakeFiles/opencv.dir/src/bitOperation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/bitOperation.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\bitOperation.cpp > CMakeFiles\opencv.dir\src\bitOperation.cpp.i

CMakeFiles/opencv.dir/src/bitOperation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/bitOperation.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\bitOperation.cpp -o CMakeFiles\opencv.dir\src\bitOperation.cpp.s

CMakeFiles/opencv.dir/src/eigen.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/eigen.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/eigen.cpp.obj: src/eigen.cpp
CMakeFiles/opencv.dir/src/eigen.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/opencv.dir/src/eigen.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/eigen.cpp.obj -MF CMakeFiles\opencv.dir\src\eigen.cpp.obj.d -o CMakeFiles\opencv.dir\src\eigen.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\eigen.cpp

CMakeFiles/opencv.dir/src/eigen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/eigen.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\eigen.cpp > CMakeFiles\opencv.dir\src\eigen.cpp.i

CMakeFiles/opencv.dir/src/eigen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/eigen.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\eigen.cpp -o CMakeFiles\opencv.dir\src\eigen.cpp.s

CMakeFiles/opencv.dir/src/general.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/general.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/general.cpp.obj: src/general.cpp
CMakeFiles/opencv.dir/src/general.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/opencv.dir/src/general.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/general.cpp.obj -MF CMakeFiles\opencv.dir\src\general.cpp.obj.d -o CMakeFiles\opencv.dir\src\general.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\general.cpp

CMakeFiles/opencv.dir/src/general.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/general.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\general.cpp > CMakeFiles\opencv.dir\src\general.cpp.i

CMakeFiles/opencv.dir/src/general.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/general.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\general.cpp -o CMakeFiles\opencv.dir\src\general.cpp.s

CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj: src/grayLevelTransform.cpp
CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj -MF CMakeFiles\opencv.dir\src\grayLevelTransform.cpp.obj.d -o CMakeFiles\opencv.dir\src\grayLevelTransform.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\grayLevelTransform.cpp

CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\grayLevelTransform.cpp > CMakeFiles\opencv.dir\src\grayLevelTransform.cpp.i

CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\grayLevelTransform.cpp -o CMakeFiles\opencv.dir\src\grayLevelTransform.cpp.s

CMakeFiles/opencv.dir/src/imageOperation.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/imageOperation.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/imageOperation.cpp.obj: src/imageOperation.cpp
CMakeFiles/opencv.dir/src/imageOperation.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/opencv.dir/src/imageOperation.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/imageOperation.cpp.obj -MF CMakeFiles\opencv.dir\src\imageOperation.cpp.obj.d -o CMakeFiles\opencv.dir\src\imageOperation.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\imageOperation.cpp

CMakeFiles/opencv.dir/src/imageOperation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/imageOperation.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\imageOperation.cpp > CMakeFiles\opencv.dir\src\imageOperation.cpp.i

CMakeFiles/opencv.dir/src/imageOperation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/imageOperation.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\imageOperation.cpp -o CMakeFiles\opencv.dir\src\imageOperation.cpp.s

CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj: src/linearInterpolation.cpp
CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj -MF CMakeFiles\opencv.dir\src\linearInterpolation.cpp.obj.d -o CMakeFiles\opencv.dir\src\linearInterpolation.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\linearInterpolation.cpp

CMakeFiles/opencv.dir/src/linearInterpolation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/linearInterpolation.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\linearInterpolation.cpp > CMakeFiles\opencv.dir\src\linearInterpolation.cpp.i

CMakeFiles/opencv.dir/src/linearInterpolation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/linearInterpolation.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\linearInterpolation.cpp -o CMakeFiles\opencv.dir\src\linearInterpolation.cpp.s

CMakeFiles/opencv.dir/src/main.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/main.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/main.cpp.obj: src/main.cpp
CMakeFiles/opencv.dir/src/main.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/opencv.dir/src/main.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/main.cpp.obj -MF CMakeFiles\opencv.dir\src\main.cpp.obj.d -o CMakeFiles\opencv.dir\src\main.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\main.cpp

CMakeFiles/opencv.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/main.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\main.cpp > CMakeFiles\opencv.dir\src\main.cpp.i

CMakeFiles/opencv.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/main.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\main.cpp -o CMakeFiles\opencv.dir\src\main.cpp.s

CMakeFiles/opencv.dir/src/noise.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/noise.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/noise.cpp.obj: src/noise.cpp
CMakeFiles/opencv.dir/src/noise.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/opencv.dir/src/noise.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/noise.cpp.obj -MF CMakeFiles\opencv.dir\src\noise.cpp.obj.d -o CMakeFiles\opencv.dir\src\noise.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\noise.cpp

CMakeFiles/opencv.dir/src/noise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/noise.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\noise.cpp > CMakeFiles\opencv.dir\src\noise.cpp.i

CMakeFiles/opencv.dir/src/noise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/noise.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\noise.cpp -o CMakeFiles\opencv.dir\src\noise.cpp.s

CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj: src/someSuperApplication.cpp
CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj -MF CMakeFiles\opencv.dir\src\someSuperApplication.cpp.obj.d -o CMakeFiles\opencv.dir\src\someSuperApplication.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\someSuperApplication.cpp

CMakeFiles/opencv.dir/src/someSuperApplication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/someSuperApplication.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\someSuperApplication.cpp > CMakeFiles\opencv.dir\src\someSuperApplication.cpp.i

CMakeFiles/opencv.dir/src/someSuperApplication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/someSuperApplication.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\someSuperApplication.cpp -o CMakeFiles\opencv.dir\src\someSuperApplication.cpp.s

CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj: CMakeFiles/opencv.dir/flags.make
CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj: CMakeFiles/opencv.dir/includes_CXX.rsp
CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj: src/transformUsedAffineMatrix.cpp
CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj: CMakeFiles/opencv.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj -MF CMakeFiles\opencv.dir\src\transformUsedAffineMatrix.cpp.obj.d -o CMakeFiles\opencv.dir\src\transformUsedAffineMatrix.cpp.obj -c D:\development_code_2022-9-10\vscode\opencv\src\transformUsedAffineMatrix.cpp

CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.i"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\development_code_2022-9-10\vscode\opencv\src\transformUsedAffineMatrix.cpp > CMakeFiles\opencv.dir\src\transformUsedAffineMatrix.cpp.i

CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.s"
	D:\mingw64-posix\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\development_code_2022-9-10\vscode\opencv\src\transformUsedAffineMatrix.cpp -o CMakeFiles\opencv.dir\src\transformUsedAffineMatrix.cpp.s

# Object files for target opencv
opencv_OBJECTS = \
"CMakeFiles/opencv.dir/src/bitOperation.cpp.obj" \
"CMakeFiles/opencv.dir/src/eigen.cpp.obj" \
"CMakeFiles/opencv.dir/src/general.cpp.obj" \
"CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj" \
"CMakeFiles/opencv.dir/src/imageOperation.cpp.obj" \
"CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj" \
"CMakeFiles/opencv.dir/src/main.cpp.obj" \
"CMakeFiles/opencv.dir/src/noise.cpp.obj" \
"CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj" \
"CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj"

# External object files for target opencv
opencv_EXTERNAL_OBJECTS =

opencv.exe: CMakeFiles/opencv.dir/src/bitOperation.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/eigen.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/general.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/grayLevelTransform.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/imageOperation.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/linearInterpolation.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/main.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/noise.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/someSuperApplication.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/src/transformUsedAffineMatrix.cpp.obj
opencv.exe: CMakeFiles/opencv.dir/build.make
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_img_hash460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: D:/development_app2/opencv/build/install/x64/mingw/lib/libopencv_world460.dll.a
opencv.exe: CMakeFiles/opencv.dir/linklibs.rsp
opencv.exe: CMakeFiles/opencv.dir/objects1.rsp
opencv.exe: CMakeFiles/opencv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\development_code_2022-9-10\vscode\opencv\CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable opencv.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\opencv.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv.dir/build: opencv.exe
.PHONY : CMakeFiles/opencv.dir/build

CMakeFiles/opencv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\opencv.dir\cmake_clean.cmake
.PHONY : CMakeFiles/opencv.dir/clean

CMakeFiles/opencv.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\development_code_2022-9-10\vscode\opencv D:\development_code_2022-9-10\vscode\opencv D:\development_code_2022-9-10\vscode\opencv D:\development_code_2022-9-10\vscode\opencv D:\development_code_2022-9-10\vscode\opencv\CMakeFiles\opencv.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/opencv.dir/depend
