#include "../include/linearInterpolation.h"



// we will define some function about image zoom used c++, 
// involved linear interpolation, binary linear interpolation, three linear interpolation.
// we will define the method for gray image first. the method as follow is not suitable for
// color images.
// first, you should define a function that can get the element about one coordinates. it means 
// give x and y, this function should return the corresponding element.
// the grayscale is range from 0 to 255, so we should define the unsigned char to accept each element in picture.
// unsigned char type can store the number that range from 0 to 255.

uchar get_scale_value(Mat &input_image, int x, int y) 
{
    uchar *p = input_image.ptr<uchar>(x);
    return p[y];
}

// then we should define the image zoom function, we will define a simple method that is named linear interpolation.
// any +-*/ float/double = double
// int +-*/ float = double
// int +-*/ double = double
// char +-*/ int = int
// int / int = int
// notice, you should ensure the grayscale of input_image is range from 0 to 255, it means the gray image downloaded
// from network is not ensure it. so you shoud use cvtColor function to get the gray image.
// float/double  modify  int = double. you can use any data type to accept the result, because the result will
// implict type conversion.
Mat scale(Mat &input_image, int height, int width) 
{
    // CV_8UC1 means gray image, 8 bits and single channel
    // 8 means 8 bits, uc means unsigned char, 1 means 1 channel. it means we will create a gray picture.
    Mat output_image(height, width, CV_8UC1);
    output_image.setTo(0); // init all element used 0
    float scale_rate_h = (float)input_image.rows / height;// calculate the height rate
    float scale_rate_w = (float)input_image.cols / width;// calcute the width rate
    // traverse each element used traditional method for circle.
    for (int i = 0; i < height; i++)
    {
        uchar *p = output_image.ptr<uchar>(i);
        for (int j = 0; j < width; j++)
        {
            // you should calculate the coordinates of the adjacent based on x, y and rate 
            // during the period of traversing.
            // of course the result will be float, but we want to get a integer, so we should casts
            // from float to int. take down the whole will happen after casts from float to int.
            // float * int = int;
            int scale_i = scale_rate_h * i;
            int scale_j = scale_rate_w * j;
            p[j] = get_scale_value(input_image, scale_i, scale_j);
        }
    }
    return output_image;
}


// of course, we can define the scale function used binary linear interpolation method.
// this method just like the follow.
/*
-->
|

i, j                   i+1, j       
       i+u, j+v

i, j+1                 i+1, j+1
0 < u, v < 1.
because the i+u, j+v what elemet we want to calculate not neccessarily at the center of the image array.
but we must could find the latest four elements. then we can calculate the result based the four elments.
f(i+u, j+v) = (1-u)*(1-v)*f(i, j) + (1-u)*v*f(i,j+1) + u*(1-v)*f(i+1,j) + u*v*f(i+1,j+1)
the f(i+u,j+v) is the result we want to get. then we start to define this function.
notice, input corrdinates are not the int. but is the float.
*/
uchar get_scale_value_binary(Mat &input_image, float _i, float _j) 
{
    int i = _i;
    int j = _j;
    float u = _i - i;
    float v = _j - j;

    // handle the border problem
    if ((i + 1 >= input_image.rows) || (j + 1 >= input_image.cols))
    {
        uchar *p = input_image.ptr<uchar>(i);
        return p[j];
    }
    uchar *p = input_image.ptr<uchar>(i);
    uchar x1 = p[j]; // f(i, j)
    uchar x2 = p[j + 1]; // f(i, j + 1);
    p = input_image.ptr<uchar>(i + 1);
    uchar x3 = p[j]; // f(i+1, j)
    uchar x4 = p[j + 1]; // f(i+1, j+1)
    return (1 - u) * (1 - v) * x1 + (1 - u) * v * x2 + u * (1 - v) * x3 + u * v * x4;
}

// then we will define the function that generate the image after interpolating.
Mat binary_linear_scale(Mat &input_image, int height, int width) 
{
    Mat output_image(height, width, CV_8UC1);
    output_image.setTo(0); // init all element used 0
    float scale_rate_h = (float)input_image.rows / height;// calculate the height rate
    float scale_rate_w = (float)input_image.cols / width;// calcute the width rate
    for (int i = 0; i < height; i++)
    {
        uchar *p = output_image.ptr<uchar>(i);
        for (int j = 0; j < width; j++)
        {
            // you should calculate the coordinates of the adjacent based on x, y and rate 
            // during the period of traversing.
            // of course the result will be float, but we want to get a integer, so we should casts
            // from float to int. take down the whole will happen after casts from float to int.
            // float * int = int;
            float scale_i = scale_rate_h * i;
            float scale_j = scale_rate_w * j;
            p[j] = get_scale_value_binary(input_image, scale_i, scale_j);
        }
    }
    return output_image;
}


// then we will learn the three linear interpolation. it means we should get the nearest coordinates based on 16 near coordinates.
// binary linear interpolation is based on four near coordinates.
// you should known that one image array is start from 0, 0 where location at the upper left corner of the image array. 
// it is a Mat variable in opencv. so the eight near coordinates is just like as follow.
/*
i,j                i,j+1
       i+u,j+v

i+1,j              i+1,j+1
it is four near coodinates above. then we should show 16 near coordinated based on it.
the difference between binary linear interpolation and three linear interpolation is the former is dedicated to an array, 
and the last is dedicated to a three dimension. just like the information above, if you add the third dimension for it.
you can find the extra 12 near coordinates. then you will have eight near coordinates. just like the information as follow

    
            i,j,z                      i,j+1,z

                i+u,j+v,z+x


    i+1,j,z                     i+1,j+1,z

but the information about three linear interpolation is not easy to show. so we will give up to show it in text file.



then we should show the information about pathways what is the shortest route from element p to q based on the adjacency and V{2, 3, 4}.
just like the information as follow. the pathways must through based on the rule about adjacency.
3   4   1   2   0       a   b   c   d   e
0   1   0   4   2       f   g   h   i   q
2   2   3   1   4       k   l   m   n   o
3   0   4   2   1       p   z   r   s   t
1   2   0   3   4       u   v   w   x   y
p(3,0) = 3 --> q(1,4) = 2
the shortest four pathways based on four adjacency, the each element that pathways through must be the four adjacency for p and q. and 
q is the four adjacency of p. and you can just up or down if you used four adjacency.
    p(3,0) = 3 -up-> 2 -right-> 2 -right-> 3 -down-> 4 -right-> 2 -down-> 3 -right-> 4=array(4,4)
    this pathways is can not arrive q, so the four adjacency is not exists for this array image.

the shortest eight pathways based on eight adjacency, you can up, down, upper right, upper left, lower right and lower left, but the rule
must be suitable for the eight adjacency. you can select the shortest route from up, down, left, right and upper right, upper left, lower
right, lower left.
    p(3,0) = 3 -upper right-> 2 -right-> 3 -upper right-> 4 -right-> 2 = q(1,4)
    the shortest route is four, it means the eight pathways is four.

then, we will consider m pathways, we should known what is the m adjacency. just like the information as follow.

V{1}

0   1   1                   a   b   c
0   1   0                   d   e   f           
0   0   1                   g   h   i
the original image          


the center elemnt is p(1,1) = 1;
all four adjacency element for p(1,1) is array(0,1)=1
all eight adjacency element for p(1,1) is array(0,1)=1, array(2,0)=1, array(2,2)=1
all m adjacency element for p(1,1) is array(0,1)=1, array(2,2)=1. notice, we should consider the N4(p)∩N4(q)∉V
just like N4(e) = bdhf, the ND(e) = acgi, the N8(e) = N4(e) + Nd(e) = ABCDFGHI, these are neighborhood.
if you consider the four adjacency or eight adjacency. you should add the condition that the element should belong the V. if it is not
meet, you should delete the element as the adjacency of e.
if you consider the m adjacency. you should add the extra condition N4(each element for N4(e) or ND(e))∩N4(e)∉V 
just like the case above.
N4(e) = bdhf, the ND(e) = acgi, the N8(e) = N4(e) + Nd(e) = ABCDFGHI
the four adjacency of e is b, because dhf∉V{1}
the eight adjacency of e is bci
the m adjacency of e is bi, because N4(c)∩N4(e) = bf, and b=1∈V, N4(b)∩N4(e) = 0∉V， and N4(i)∩N4(e)=fh, and fh∉V
you should have known the difference between eight adjacency and m adjacency, the last is more strict and it can handle the ambiguity of the eight adjacency.


then we should give the m pathways based on the former case.
our pupose is from p to q, and give the shortest route based on the four adjacency, eight adjacency and m adjacency.
N4(p)=kz    
four adjacency of p is k    
N4(k)=fl
four adjacency of k is l
N4(l)=kgmz
four adjacency of l is km, k is ignored
N4(m)=hlrn
four adjacency of m is lr, l is ignored
N4(r) = mzws
four adjacency of r is ms, m is ignored
N4(s) = nrxt
four adjacency of s is rx, r is ignored
N4(x) = swy
four adjacency of x is sy, s is ignored.
N4(y) = xt
four adjacency of y is x, x is ignored.
util here, the route is closed, so the four pathways from p to q for this image is not exists.


then we can consider the eight pathways.
N8(p) = klzvu, eight adjacency of p is klv. v is ignored, we can choose k or l as the next step. becaused of the shortest route, we should choose l.
N8(l) = fghmrzpk, eight adjacency of l is mrpk, rpk is ignored because of the shortest route is needed.
N8(m) = ghinsrzk, eight adjacency of m is isrl, srl is ignored because of the shortest route is needed.
N8(i) = cdeqonmh, eight adjacency of i is dqom, dom is ignored because of the shortest route is needed.
so the eight pathways is from p>>l>>m>>i>>q, the length is equal to four.

then we can consider the m pathways based on the eight pathways.
N8(p) = klzvu, eight adjacency of p is klv. the m adjacency of p is k.we should add the extra condition, N4(p)∩N4(k)=0∉V, N4(p)∩N4(l)=kz, and k=2∈V, N4(p)∩N4(v)=zu∉V,
    the l is not suitable for the rule of m adjacency. the v is ignored because of the shrtest route is needed, so you can just step to k.
N8(k) = fglzp, eight adjacency of k is lp, the m adjacency of k is l.N4(k)∩N4(l)=0∉V, and p is ignored. so you can just step to l.
N8(l) = fghmrzpk, eight adjacency of l is mrpk. the m adjacency of l is m. pk is ignored because it will be return. r is ignored because of the shortest route even if we have not
    judge if it is suitable for the rule of m adjacency. the m is be left.N4(l)∩N4(m)=0∉V, so you can just step to m.
N8(m) = ghinsrzk, eight adjacency of m is isrl, the m adjacency of m is i, srl is ignored because it will be return, we just need to adjust
    the suitablely of i for the m adjacency. N4(i)∩N(m)=hn∉V. so you can just step to i.
the last step, you should step to q directly.
then, so the m adjacency route from p to q is p>>k>>l>>m>>i>>q, the length is equal to five, it is bigger than the route length of eight pathways.

connectivity.
if p and q is belonged to s what an element subset of an image array. if exist a pathways that from p to q can connect all element in s.
we can conclude that p and q is connected in s. notice, this saying just applied to s set.
the connectivity can be divided into three type, just like 4-connectivity, 8-connectivity and m-connectivity based on the pathways method above.
connected component, the element set that can connect to p in s is named the connected component of p. just like it may have multiple
sets that can connect to p in s. so the elements in each pathways is a set what is named the connected component of p.
if it just has one connected component in s, then we can name s as connected set. it means a connected set just has one connected component.

regions
if r is a connected set, then r is a region of image. the defination of region is just based on four adjacency or eight adjacency. not the m adjacency.
if two regions r1 and r2. if r1 union r2 is a new connected set, so it can be named r1 and r2 is adjacency. it is the adjacency about regions.
notice, the region just involved four adjacency or eight adjacency, not has the m adjacency, it is the difference between region adjacency
and element adjacency. 

boundary
assume that it has k numbers regions that are not adjacency, 
*/