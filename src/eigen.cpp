#include "../include/eigen.h"

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
    cout << endl;
    cout << "a 3*5 zero matrix\n" << matrix2 << endl;
    cout << endl;
    cout << "a 4*4 zero matrix\n" << matrix3 << endl;
    cout << endl;

    // define unit matrix.

    // define the other specific matrix.
}