#ifndef _EIGEN_H
#define _EIGEN_H
#include "general.h"
#if ISOPENEIGEN
// some basic operation about Matrix in Eigen.
// involved some operation of VectorXd, Vector4d, MatrixXd, and Matrix4d.
void defineSpecificMatrix();
// then, we will define how to transform between Matrix and Mat
void transform2Matrix(Mat inputImage, MatrixXd &Matrix);

#endif

#endif