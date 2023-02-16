#include "../include/eigen.h"

#if 0
void defineSpecificMatrix() 
{
    // define a general matrix
    Matrix<int, 3, 3> matrix1{
    {1, 2, 3}, 
    {4, 5, 6},
    {7, 8, 9}
    };
    cout << "a general matrix\n" << matrix1 << endl;

    // define a zero matrix
    // define a m*n zero matrix
    MatrixXd matrix2 = MatrixXd::Zero(3, 5);
    // define a 4*4 matrix
    Matrix4d matrix3 = Matrix4d::Zero();

    Matrix4d matrix4 = Matrix4d::Ones();
    cout << endl;
    cout << "a 3*5 zero matrix\n" << matrix2 << endl;
    cout << endl;
    cout << "a 4*4 zero matrix\n" << matrix3 << endl;
    cout << endl;

    // some operation, notice, you can nou use matrix
    // + int. but you can add, minus, multi, divide two matrix.
    MatrixXd test = matrix3 + matrix4;
    cout << test << endl;

    Matrix4d matrix5 = matrix3 * matrix4;
    cout << matrix5 << endl;

    // define a vector used vectorXd
    // you can multi the matrix(4*4) and vector(4*1) directly.
    // but you can not multi the matrix(4*4) and matrix(4*1) directly.
    VectorXd vector1(4);
    vector1 << 1, 2, 3, 4;
    // or you can define it use this method.
    // VectorXf vector{{1.1, 1.2, 1.3, 1.4}};
    // the default vector size is m*1, it is a columns vector.
    cout << matrix4 * vector1 << endl;
    // these are matrix multi, add, minus.
    // this object can be a matrix, vector and a scalar.
    // just like matirx + scalar, matrix - scalar, * and divide.
    // vector + vector, vector - vector, multi and divide.
    // matrix + matrix, matrix - matrix, matrix * matrix.
    // matrix * vector.
    // you should notice, the multi and divide between matrix and matrix,
    // matrix and vector should conform to the rule about multi. 
    // otherwise, you will get error when you compile you program used gcc.

    // but how to do if you want to dot two matrix?
    // two matrix should be the same dimension, just like 4*4 and 4*4
    // you want to dot multi the two matrix.
    // you can define a matrix use MatrixXf, Matrix4f, Matrix<float, m, n>
    // to define the matrix object.
    // Xf means you can define any dimension that store the float data.
    // 4f means you can define a 4*4 dimension that store the float data.
    // Matrix<float, m, n> means you can define any dimension that store the float data.
    // MatrixXd mean any dimension that can store the double data.
    MatrixXd matrix7 {
        {1.1, 1.2, 1.3, 1.4},
        {2.1, 2.2, 2.3, 2.4},
        {3.1, 3.2, 3.3, 3.4},
        {4.1, 4.2, 4.3, 4.4}
    };
    Matrix4d matrix8 = Matrix4d::Random();
    // dot two Matrix, notice it is difference from the operation multi.
    // if you want to dot multi, you should use the array method
    // for Matrix to calculate the expression.
    // this method is used for a matirx add, minus, divide and multi
    // one scalar.
    cout << matrix8 << endl;
    MatrixXd result = matrix7 * matrix8;
    MatrixXd result1 = matrix7.array() * matrix8.array();
    cout << result << endl;
    cout << result1 << endl;
    // define unit matrix.
    
    // define the other specific matrix.

    // you can also transform the data type, we should consider it about
    // eigen and opencv. because the data type transform can not be implicit conversion.
    // transform from MatrixXd to MatrixXf, means from double to float.
    // you can also use this method in opencv
    // just like, matf.convertTo(matd, cv_64F)
    MatrixXf transformMatrix = matrix7.cast<float>();
    cout << transformMatrix << endl;
    cout << transformMatrix.rows() << ", " << transformMatrix.cols() << endl;
}


void transform2Matrix(Mat inputImage, MatrixXi &matrix)
{
    cv2eigen(inputImage, matrix);
}

#endif