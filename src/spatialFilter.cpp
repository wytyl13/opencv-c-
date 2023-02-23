/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-02-03 13:51:42
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-02-03 13:51:42
 * @Description: this file is about anthor image enhancement method, spatial
 * filter. it is a device, you can also name it used spatial filter device.
 * it is widely used in various image processing aspect. we will just consider the
 * application in image enhancement.
 * the image enhancement method that filter by changing or inhibition of the frenquency
 * component of the image. the filter device has low frequency and high frequency.
 * the application of low frequency is to smooth the image by blurred image.
 * and the spatial filter device can smooth the image directly on the image itself.
 * this is a spatial domain handle method.
 * the similarly, spatial filter handle is also change the gray value directly used 
 * the cooridinates function value or the neibohood of the center coordinates function value.
 * this changed concept is similar to the former gray vlaue transformation. they are
 * all changed the gray value directly on the image itself and the mapping value
 * is the corresponding funtion value. they are all transformation in the spatial domain.
 * of course, you can also name the filter used kernel. this kernel is defined by ourselves.
 * the kernel in local transformation is scaned matrix function, each element in it
 * is determined by the orginal vlaue. but in spatial filter device this kernel is defined
 * by ourseleves, it will multiply the scaned matrix, it means you will multioly two
 * matrix, one is scaned matrix from the original image, one is the filter device kernel
 * that we defined. it may be not fit that we have name the scaned matrix used kernel
 * in local transformation, it is because we have learned deep_learning before learning
 * the digital image processing. so we habitual defined the scaned matrix as kernel.
 * then, we have figure that the difference between scaned matrix and kernel. the former used
 * original image value, the last is defined by ourselves. and the kernel will
 * multiply with the orginal scaned matrix. so we will separate the scaned matrix and kernel
 * in last note.
 * 
 * the spatial filter device involved linear and nonlinear. 
 * the difference between local histogram transformation and spatial filter device
 * is the former just scaned the original image, and the mapping value will be calculated
 * used orginal miltiply a constant, or the mean of scaned matrix, or the center of the scaned matrix
 * mapping value that based on the equalization transformation for the scaned matrix, or
 * based on the matching transformation for the scaned matrix. so you can find the gray value transformation
 * that based on the histogram local transformation are all mapping the original value
 * based on the orginal scaned matrix indicators. but the spatial filter device
 * will add a kernel matrix that be defined by ourselves to influence the original gray value.
 * multiply the original scaned matrix by the kernel. we will get more powerful image 
 * transformation result. the convolution in deep learning is also used the concept. of course the digital
 * image processing will also use the concept of convolution.
 * 
 * what we consider is m*n size kernel, m = 2a+1, n = 2b+1, m and n are all odd number. the m is the 
 * row number in kernel, and n is the column numbers in kernel. the length of the kernel is (m-1, n-1)
 * the linear spatial filter that m*n size kernel to M*N original image can show used the follow expression
 * we should image the shape and location that the kernel corresponding to the origianl image.
 * the center coordinated of kernel is 0,0, and the origin point of the original image is 0,0
 * so you can find the relationship that the left upper scaned matrix corresponing to the kernel.
 * scaned the kernel size from the origin point, and the origin point of the kernel is the center point.
 * the original element use f(x, y) represent. and coefficient of the kernel used w(s, t) represent.
 * the size of kernel is m*n, and m = 2a+1, n = 2b+1
 * then the scaned matrix multiply the kernel is euqal to the expression as follow. x, y is each element for
 * the matrix. w(s, t) is each coefficient of the kernel. g(x, y) is the result. the x and y is fixed coordinates
 * what we want to get is the g(x, y), it is the gray value we want to mapping the f(x, y). so you will get
 * the aim about spatial filter transformation. it is find the mapping value for f(x, y) what gray value in 
 * original image about the x, y coordinates. coordinates x, y in orginal image is corresponding to the origin
 * point in kernel matrix. so just you can corresponding the w(s, t) and f(x+s, y+t)
 * and the range -a to a and -b to b menas accumulative each element of the (2a+1 * 2b+1) kernel matrix 
 * multiply each element in scaned original matrix. then change first x, y coordinates in the original image
 * used g(x, y). then ++x, y coordinates, make the center of the kernel can access each element in original image.
 * g(x, y) = Σs=-a_aΣt=-b_b w(s, t)*f(x+s, y+t)
 * why use the center of the kernel matrix as the origin point? in order to simple the expression.
 * 
 * 
 * the related calculation process as follow.
 * move the center of the kernel in the original image, and calculate the sum of each element. this is the simple
 * spatial filter kernel. then we will introduce the spatial convolution based on the spatial filter device.
 * util here, you can image the kernel used a two dimension array, it just like a table. the former problem about
 * the size and length are all dropped. because the size is equal to length if you image the picture is a table.
 * 
 * the kernel convolution and the spatial filter device is different but corresponding.
 * 
 * the spatial convolution is similar to the general filter, the difference between them is 
 * the spatial convolution will rotation the kernel 180 degrees. it means you will get the same result
 * if the value in kernel is symmetrical about the center of the kernel, of course, the center of the kernle
 * is also the original of the kernel.
 * 
 * then, we will name the spatial filter operation as spatial correlation, and the spatial convolution operation
 * as spatial convolution, then, we will discuss the difference between spatial correlation and spatial convolution.
 * notice, the original point of one image is located at the leftUpper of the image.
 * the original point of one kernel is located at the center of the kernel. 
 * the cause of processing it is because you will get the simple expression based on this definition.
 * 
 * then, we should consider the application impulse in digital image processing.
 * unit impulse, you can image that a crest in one function. it can be range from infinity.
 * δ(t) = (t == 0) : infinity ? 0
 * this expression is about the unit impulse of continuous variable t in the place of t = 0.
 * δ(t) meet the integral expression: ∫_infinity δ(t)dt = 1.
 * if t means time, you can image that one impulse is a peak signal that with the infinity amplitude and
 * the duration is zero.
 * 
 * ∫_infinity f(t) * δ(t - t_0)dt = f(t_0).
 * ∫_infinity f(t) * δ(t)dt = f(0) if t_0 = 0.
 * 
 * it is the continuous impulse, another is discrete impulse.
 * unit discrete impulse can show as follow
 * δ(x) = (x == 0) : 1 ? 0. 
 * δ(x) meet the expression Σ_infinity δ(x) = 1.
 * Σ_infinity f(x) * δ(x - x_0) = f(x0).
 * 
 * the discrete impulse function of coordinates(x_0, y_0) can be shown used the expression as follow.
 * δ(x - x_0, y - y_0) = ((x_0 == x) && (y_0 == y)) : A ? 0
 * for one dimension array, 0 0 0 1 0 0 0 0.
 * the unit impulse coordinates is δ(x - 3), because the element value is 1 when x_0 = 3.
 * if A = 1, then it is unit impulse.
 * 
 * for two dimension matrix. 
 * 0 0 0 0 0
 * 0 0 0 0 0
 * 0 0 1 0 0
 * 0 0 0 0 0
 * 0 0 0 0 0
 * the unit impulse coordinates is δ(x - 2, y - 2).
 * more detailed about impulse we will deep learning. we just given some expression conclusion at here.
 * 
 * 
 * we usually rotated the kernel not the original image or the result image.
 * if the size of kernel is m*n, and the size of original image is M*N
 * we usually displayed the kernel used w, and displayed the original image used f.
 * the dimension of complete correlation and convolution is (M+m-1, N+n-1).
 * notice, here the filter has been similar to the kernel. you can consider them as the same word.
 * filter, or kernel. the operation about the original image f and kernel w is named convolution.
 * the original spatial filter is not the convolution. it is spatial filter, but they are all
 * do the dot between the scanned matrix and kernel or filter. the difference just is the kernel will
 * rotate 180 degrees based on the filter. so you should distinguish the kernel and filter.
 * the spatial filter and spatial convariance.
 * sometimes, we usually do many convolution operation to one image. so you should give many different 
 * kernel, and convolution it by the order. you can image this expression as follow.
 * if you have defined three kernel. w1, w2, w3.
 * the correlation result is equal to w1*w2*w3*f, and it is euqal to w*f, w = w1*w2*w3.
 * it can just be used in convolution, not used it in spatial filter. because the spatial filter is not
 * suitable for the exchange law.
 * 
 * notice, * means the filter or convolution, the filter is verb, it is not noun. you can image it has
 * the same function with convolution.
 * 
 * if each kernel has the same size m*n, we can get the size of w(w1*w2*w3)
 * we can image the normal result. f(M*N)*w1 = (M+(m-1), N+(n-1))
 * f(M*N)*w1*w2 = (M+(m-1)*2, N+(n-1)*2)
 * f(M*N)*w1*w2*w3 = (M+(m-1)*3, N+(n-1)*3)
 * then, we can get the size of w based on the result and the dimension of w1.
 * f(M, N)*w(unknow_x, unknow_y) = ((M+(unknow_x-1)), (N+(unknow_y-1))) = (M+(m-1)*3, N+(n-1)*3)
 * -> (unknow_x-1=(m-1)*3, unknow_y-1=(n-1)*3) -> unknow_x=(m-1)*3+1, unknow_y=(n-1)*3+1
 * w((m-1)*3+1, (n-1)*3+1)
 * but the result about the dimension of w is w((m-1)*3+m, (n-1)*3+n)
 * 
 * we can get the expression to inference the dimension of w. w=(w1*w2*w3..*wQ), and each w has the same dimension (m, n)
 * w = (w1*w2*...wQ), each convolution kernel w dimension is (m, n)
 * execute the convolution operation
 * the dimension of w is [(m-1)*Q+1, (n-1)*Q+1]
 * notice, the similar expression about correlation can not write. because the correlation is not suitable
 * for the exchange law.
 * 
 * but we should also consider the different shape of the kernel. we will consider it last.
 * 
 * then, we will consider a special kernel what names separable filter kernel.
 * this kernel can be expressed by the cross product of two vector. of course, the vector generally means
 * colunmn vector. vector involved row vector and column vector. the product of one colunm vector and one
 * row vecotor is the cross product, the product of one row vector and one column vector is the inner product.
 * so, if one kernel w(2, 3) is euqal to the product of one column vector(2, 1) and one row vector(1, 3), 
 * and the column vecor and row vector are all one dimension. the kernel w is named as separable filter kernel.
 * it mean it can be named as separable filter kernel if one two dimension kernel can be expressed
 * by the product of two one dimension vector or the cross product of two vector.
 * then, we can conclude that, if one separable filter kernel w = w1 * w2. 
 * w * f = (w1 * w2) * f = (w2 * w1) *f = w2 * (w1 * f) = (w1 * f) * w2
 * the above expression used the feature of convolution, exchange law and associative law. this expression is
 * suitable for the convolution. it will be not suitable if you out of the convolution, just like you use
 * the w = w1 * w2 = w2 * w1, this expression is error.
 * we can conclude from the expression, the convolution of one separable filter kernel and one image(w * f),
 * it is equal to the convolution of w1 and f, then executed the convolution of the former result w1*f and
 * w2. so it means, the convolution of w and f(w*f) is equal to the convolution of (w1*f) and w2, if 
 * w is equal to the cross product of w1 and w2.
 * what the meaningful for the separable filter kernel? you can image the computation of the generally kernel.
 * f(M*N), w(m*n), the computation of the convolution of w and f is equal to M*N*m*n
 * but if the kernel is separable filter kernel, w(m*n) = w1(m, 1) @ w2(1, n),
 * w*f = (w1*f)*w2, the computation of the convolution of w1 and f is equal to M*N*m, 
 * the computation of w1*f and w2 is equal to M*N*n, so the computation of the convolution of w and f
 * is euqal to M*N*(m+n). you can compare these two computation between general kernel and separable 
 * filter kernel.
 * [M*N*m*n] / [M*N*(m+n)] = m*n / (m+n). the difference between them are huge.
 * then, we will introduct the concept of rank in mathmatics. the rank of the result matrix is euqal to 1 if
 * the matrix is euqal to the cross product of two vector. and we have known the matrix is separable filter kernel.
 * so we can conclude whether the kernel is separable filter kernel based on the value of rank of the kernel.
 * 
 * in view of the feature of separable filter kernel. we will separate the separable filter kernel before we
 * done the convolution operation. then, we will define the function separateKernel function first.
 * 
 * we have defined these function, and update the function about can handle both the one dimension
 * kernel and two dimension kernel.
 * you may forget the process about these problem, but some content you must memory them. just like as follow.
 * no matter one dimension kernel or two dimension kernel.
 * one dimension kernel can be (1, m), (m, 1), two dimension kernel must be (m, m). you can use w(m, n) to represent
 * the kernel, and use f(M, N) to represent the image. notice the limit conditions for the dimension about w.
 * the dimension of zero padding matrix is f(M+m-1, N+n-1). the m and n must be an odd number.
 * and the original point that located at the zero padding matrix is [(m-1)/2, (n-1)/2], notice, 
 * you can use int halfRow = rows / 2, it is because int has done the implicit take down the whole.
 * but the concept of original point is [(m-1)/2, (n-1)/2]. 
 * so that's all content you must memory it.
 * and you should know that the reason we used separated filter kernel is because it is 
 * more efficient than the original kernel. and the difference in efficiency is huge.
 * 
 * that' all, then, we will consider the difference between spatial domain filter and frequency domain filter.
 * according to my current knowledge, the difference between them has two point, 
 * one is the difference between two domain. the spatial and frequency, just like we have learned in former, 
 * the frequency will not change the gray value directly.
 * one is the different calculate, just like the product and divide in spatial domain, they are all can be instead 
 * by add, minus in frequency domain, so it is more efficient in frequency domain.
 * that's all. we will deep learning the difference between them.
 * 
 * then, learn here, it is necessary to clear up the filter, convolution kernel, and operator.
 * the operator is also named the convolution kernel.
 * there are two mode, involved spatial domain and change domain.
 * the different between filter and kernel is the convolution, the former filter has not rotated 180 degrees
 * for the filter. the kernel will be rotated 180 degrees when executed the convolution operation.
 * 
 * of course, the operator and filter can all be used in two domain.
 * the different domain can be named different.
 * but the exchange gray value has two method so far, linear exchange and nonlinear exchange.
 * linear exchange we have learned, just like exchange the scanned gray value used the center of the result
 * of filtering or convoluting, mean filter is a general linear filter. the nonlinear exchange involved 
 * median filter and other.
 * 
 * notice, some filter or kernel is dedicated to the specific domain. because some filter or kernel is based
 * on the gray value. and the exchange domain usually can not get the gray value. so the kernel or filter
 * can not out of the domain. just like the sharpen kernel, it is usually dedicated to the high frequency domain,
 * so we named it as high-pass filter, util here you need not to seriously the difference between filter,
 * convolution and operator, why use kernel or operator? because it is suitable for the exchange law and
 * association law, it is more efficient, so the difference between filter, kernel, operator is the same matrix, 
 * the difference between them are how to operate these matrix, if you do convolution operation for the matrix
 * and the original image, the it is convolution kernel. or it is traditional operation. so learned here, we 
 * will do the convolution operation for all case, so you can drop the difference between them, because they are
 * all matrix, and the operation are all convolution. so we will rename these matrix used filter, it is 滤波器
 * , drop the kernel, operator. you should understand they are with the same meaning.
 * 
 * then, we will use filter to describe all the kernel, operator and filter.
 * the matrix has the specific domain, just like sharpen filter, it is usually used in high frequency, 
 * so it belong to the high-pass filter. so if you used it into the spatial domain, you will not get the
 * efficient what you want. you should use it in high frequency domain.
 * 
 * and you should know the generally filter just like median filter and mean filter belong to the spatial
 * filter. because they are operated by the gray value.
 * 
 * then, we will intoduct how to define yourself filter.
 * just like the mean filter. it is a linear spatial filter, belong to spatial domain.
 * it means get the neighborhood of the scanned coordinate in original image. and calulate the mean
 * of the them and change the gray value of scanned coordinate used the mean value. then, we can define the
 * filter used the matrix what dimension is 3*3 and the element are all 1/9. you will get the efficient what
 * you want. just like the smooth filter, SMOOTHKERNELCASSETTE we have defined it in general.h head file.
 * it is a mean filter. and the smooth kernel and fuzzy kernel has the same design, and the difference between
 * them is the size of kernel, the fuzzy has the bigger size of kernel. because it will consider the 
 * mean of bigger scanned region. just like the gaussion blur, it is the same concept but not use the
 * mean to change the gray value. it used the gaussian weighted average. but it just only can handle the
 * noise that is suitable for the gaussian distrubution, it can also be named as normal distribution.
 * 
 * you should notice, the gaussianBlur is also the gaussian denoise. and the fuzzy filter can be also
 * used for the denoise operation, because the concept of them is similar. but the distribution of
 * noise will influence the efficient of denoising. just like saltPepper noise will be treated use the specific
 * filter, and the gaussian filter can just handle the gaussian noise. because the saltPepper noise is rand generated.
 * so you should use another filter to handle, but these fuzzy, denoising, mean filter are all dedicated to
 * the spatial domain, the sharpen filter is used for the frequency domain. you should distingush them.
 * 
 * then, we will deep learning how to define the spatial filter kernel.
 * you can define the kernel based on the feature of gray value. and you can define this kernel
 * based on a two dimension function, just like the gaussian distribution function.
 * and you can define the kernel based on some software.
 * just like smmoth low-pass spatial filter.
 * the smooth spatial filter is the linear spatial filter, it means the convolution of filter kernel
 * and the image. the convolution of the smooth filter kernel and image will smooth the image and 
 * the degree of smoothing will be depending on the size of the kernel and the kernel param.
 * 
 * the cassette filter kernel, it is the most simple separable low-pass filter kernel. the kernel has
 * the same param, generally is 1, you can also define the same value is n.
 * and you should define the normalized constant, the constant is depended on the param n and the size
 * of kernel, the constant will be 1/m*m*n, if the size of kernel is (m, m). just like as follow
 * 1 1 1
 * 1 1 1 * constant(1/9)
 * 1 1 1
 * 
 * 2 2 2
 * 2 2 2 * constant(1/18)
 * 2 2 2
 * these two kernel will be same efficient. they are all smooth filter kernel, the degree of
 * smoothing will depend on the size of the kernel. bigger size will result to the more degree of fuzzying.
 * the small size will result to smooth, the big size will result to fuzzy.
 * the defects of the cassette filter kernel is directional, we usually use the filter kernel of 
 * circular symmetry. and the gaussian filter kernel is the only one separabel and 
 * the circular symmetry filter kernel. so the gaussian filter kernel is better than the cassette filter
 * kernel.
 * 
 * learned here, we should expand the high-pass and low-pass, although these two words is original from
 * the frequency domain, but we can also use it to represent some feature in spatial domain.
 * just like the high-pass means the high frequency, low-pass means the low-frequency in frequency domain.
 * the low-pass means the edge in spatial domain. these region is low frequency in spatial domain.
 * so the smooth filter kernel is low-pass filter kernel in spatial domain.
 * we have learnd one feature of gaussina filter kernel, it is the circle symmetry, the another feature
 * is the result of the product or the convolution of two gaussian function are also the gaussian function.
 * so we will need not to calculate the single one dimension convolution again, we can get the 
 * mean and standard devitation of the composite filter kernel directly.
 * you should notice, the size of gaussian filter kernel is limited by the standard devitation you have defined.
 * the limit is (6*std, 6*std), just like you have defined one gaussian filter kernel with the standard
 * devitation is 7, then the size will be limited less than (6*7, 6*7) = (43, 43)(notice, the minimize
 * odd number), it means you will get the same efficient if your size is greater than (42, 42). 
 * it means the size (42, 42) will be same efficient to the (45, 45)
 * the differece between gaussian filter kernel and cassette filter kernel is the former will result the
 * more degrees of smoothing at the edge region, the last is on the contrary, it will result the low
 * smooth at the edge region.
 * 
 * then, we will distinguish the different convolution method. we have defined the zero padding method.
 * the other methods involved the mirror padding and copy padding. the efficient will be different.
 * which padding method we should select it? it depend on the value feature of the boundary region in
 * one image. if the value in boundary region is constant, you should select copy padding.
 * if the boundary region has more details you should select the mirror padding.
 * 
 * the relationship between the attribution of kernel and the size of image. you should use
 * the same multiple increase of the kernel as the increase of the size of the image.
 * it means, if you want to get the same smooth efficient when you handled the different size
 * image, you should change the size, standard devitation of the kernel. if your image size increase 
 * four times, the size, standard devitation of the kernel should be increased four times if you want
 * to get the same efficient. so we generally use the ceiling size, just like we generally defined
 * the attribution of the kernel is (6*standard deviation + 1, standard deviation)
 * Mat gaussianKernel71 = getGaussianKernel_(7, 1);
 * Mat gaussianKernel132 = getGaussianKernel_(13, 2);
 * this is the gaussian kernel function, the first param is size of the kernel, the second 
 * param is standard diviation.
 * 
 * the application of gaussian filter kernel is generally used many scenario.
 * just like, we have learned how to reduct the noise, smooth and fuzzy image used it.
 * then, we will consider another application, shadow correction.
 * the concept of shadow correction is g(x, y) = f(x, y) * s(x, y).
 * the g(x, y) is the shadow image, the f(x, y) is the normal image, the s(x, y) is the shadow image.
 * so we can get the f(x, y) used g(x, y) / s(x, y). the g(x, y) is the inputImage. then, how to get the 
 * s(x, y), we will get the s(x, y) as the standard deviation and size of the gaussian filter kernel
 * increased lager enough. it means you can get the s(x, y) image if you convoluted the image used
 * an enough big size and standard deviation gaussian filter kernel.
 * you can find, as the size and standard deviation of the image increasing, the image will be more
 * fuzzy util you can find the shadow obviously. you can find the image has became one image just like
 * the binary image, the shadow region shows the blank, and the other regions shows white. of course, it is
 * different from the binary image. but you should know, this image is the s(x, y), it has the shadow informations
 * . you can divide it by the shadow image g(x, y). then, just like g(x, y) / s(x, y), then you can get the normal
 * image what you want. but we have failed to test it successful, so we will test it and the application of
 * threshold value limit conditions used into gaussian filter kernel in the image convolution filed.
 * 
 * then, we will consider the median filter kernel. it is the method about nonlinear filter based on the
 * statistical sorting. the median filter kernel will change the center gray value used the median of the
 * neighborhood of the center coordinate, the application of the median filter kernel involved the reduction
 * of the random noise, compared with the same size of the linear smooth filter kernel, the feature is
 * it can result the smaller degrees of smoothing. so you can use it to reduct the random noise, just like
 * the salt pepper noise, it is more efficient than the linear smooth filter kernel. the nature of the median
 * filter kernel is make the value of center coordinated closer to their neighboring points. the median filter
 * kernel is the most efficient filter in the statistic sorting filter. of course, you can also use the maximun
 * of the neighboring of the center to change the center gray vlaue. you can also use the minimize value to
 * instead it. it will be named as maximun filter kernel and minimize filter kernel. then, we will define 
 * the function medianFilter. because the median filter need not to used the convolution and kernel.
 * so we will create the separate function.
 * 
 * learned here, we will start the high-pass filter, sharpen filter kernel. this filter will pass the
 * high frequency and inhibition of the low frequency. the shrpen filter kernel can be established based
 * on the first derivative and the second derivative. the first derivative and the second derivative of
 * the constant gray value region must be zero. because the derivative show the rate of changing.
 * you should distingush the slope and the step. they are all the special situation.
 * the first derivative of the slope must be nonzero, because the gray value is continuous, and
 * the second derivative of the slope must be zero, because the rate of changing is constant.
 * the second derivative of the start or end of the slope and step must be nonzero. 
 * and the first derivative of the start of the slope or the step must be nonzero.
 * the first derivative of the end of the slope and step must be zero if connected with the constant region.
 * because the gray value will be changed. you can draw the figure of the derivative about the constant, 
 * slope and step. you can find some new feature about the digital image. it will be amazing.
 * what we should notice is, the slope is close to the edge in the image. because the gray value is continuous
 * in the edge, so the first derivative of the edge is nonzero. so the first derivative of the edge will
 * generate the widely edge, why? we can give a case. just like as follow.
 * 1 1 1 1 1 2 3 4 5 6 8 8 8 8 8  -> gray value.
 * 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0  -> first derivative.
 * 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0  -> second derivative.
 * the first derivative of the start or end of the slope and step must be nonzero.
 * the first derivative of the start of the slope and step must be nonzero. it means the end can be zero.
 * of course. the first derivative of the end of the slope and step will be zero if they connected with
 * a contant region. we can find it is suitable for these concept above case.
 * we can find the first derivative result to the slope what means the edge in digital image has
 * the widely region, but the second derivative has a narrow region. so we will compare the features between these two
 * derivative. we can intuitively find that the second derivative will result to more clearly edge, it has 
 * generated the edge that width is 1 pixel and cutted by two zero. and the amount of the calculation about
 * the derivative is smaller than first derivative obviously. so the second derivative will be a very good tool
 * to sharpen the image. then, we will learn how to sharpen the image used the second derivative of the image.
 * then, we have update the function spatialFilterOperation, this function has had the sharpen function used
 * four laplasian operators. the difference between the laplacian operator and other filter kernel is
 * the other can use the linear scaling function, but the laplacian need to use the truncation mapping method
 * , just like gray value = (gray value > 255, < 0) : (255, 0) ? (gray value). and the laplacian operators
 * are all the symmetric square, and the sum of all gray value in these two square are all zero. you can find
 * the rule based on them. so we have defined the generally function based on these feature of these operator.
 * you can find the laplacian operator convolution with the original image result to the edge image what the
 * edge region is the original value, and the weakening the other region. then you can add the convolution image
 * and the original image. you will get the image than enhanced the edge what is the image of sharpening.
 * 
 * then we will create another file what dedicated to the frequency domain filter.
***********************************************************************/
#include "../include/spatialFilter.h"


/**
 * @Author: weiyutao
 * @Date: 2023-02-14 16:14:41
 * @Parameters: 
 * @Return: 
 * @Description: this function involved some official function about spatial filter.
 * Mat is the original data type in opencv, Mat_ is the super data type based on Mat.
 * and you can accept the Mat_ used Mat. it is equal to the pilymorphism of class.
 */
void officialFilterTest(Mat &inputImage, Mat &outputImage, Mat kernel) 
{
    if (kernel.empty())
    {
        sys_error("the kernel you passed is invalid");
    }
    // define the kernel, this kernel can sharpen the image.
    // this Mat_ is a super application for Mat class. you can use it more convinence.
    filter2D(inputImage, outputImage, inputImage.depth(), kernel);
}

void officialImageMixTest(Mat &inputImage1, Mat &inputImage2, Mat &outputImage, float firstWeight) 
{
    // ensure the same size of two image
    if (inputImage1.rows != inputImage2.rows || inputImage1.cols != inputImage2.cols || inputImage1.type() != inputImage2.type())
    {
        resize(inputImage1, inputImage1, Size(300, 300));
        resize(inputImage2, inputImage2, Size(300, 300));
    }
    addWeighted(inputImage1, firstWeight, inputImage2, 1 - firstWeight, 0.0, outputImage);
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-20 17:17:56
 * @Parameters: you should pass the filter kernel, the different kernel will get 
 * the different efficient. the kernel generally is 3*3 mat. just like fuzzy and sharpen. of course, you should
 * also pass the inputImage and outgoing image. you will get the smaller wasted for memory if you use reference
 * symbol to modify the parameters. because the stack will not create the new container to store the parameter
 * you shold input one gray image, or you will get error. because we have defined the mapping function based on the 
 * gray image.
 * @Return: outputImage, the image you interested in.
 * @Description: 
 * spatial filter function created by ourselves test.
 * you should consider your mat data type if you want to dot or @ two mat in opencv.
 * 8UC1 must be error if you do matrix operation. opencv have overloaded the operator *
 * matrix1 * matrix2 means two matrix multiplication. if you want to get the inner product
 * of two mat, you should use matrix1.dot(matrix2).
 * 
 * notice minMaxLoc is strict for the channels of one image. and the different type image will
 * get the error. and notice, the filter result can not influence next filter result. it means you should
 * create one new image and change the corresponding coordinates in the new image.
 * 
 * the spatial filter means the value of coordinates in the original image will be changed. it means you will
 * change the gray value based on the original image what is the spatial domain. this changed method can be
 * also named as high-pass filter. it is corresponding to the low-pass filter. the low-pass filter will 
 * change the gray value in transformation domain. just like the frequency domain. the filter device in
 * low frequency is named as low-pass filter. you should also distinguish the linear filter and nonlinear filter.
 * 
 * 
 * 
 * notice, the spatial correlation operation should range the whole image. so you should zero padding the original
 * image first. then scanned from (0,0) to (rows, cols).
 * Rect(Point(x, y), width, height);
 * Size(width, height);
 * notice the application of copyTo function in spatial filter and spatial convolution. it can simple to 
 * apply to the zero padding. because you used the shallow copy. if you used deep copy, you will get the
 * complex operation.
 * 
 * then, we will update this function, it will add the function that can handle both the 1 dimension kernel
 * and two dimension kernel.
 * 
 * then, this function have the function that can handle the one dimension kernel and two dimension kernel.
 * you can call it in the function spatialFilterUsedSeparatedKernel. we have defined the function getGaussianKernel
 * function.
 */
void spatialFilterOperation(Mat &inputImage, Mat &outputImage, Mat kernel) 
{
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;
    if (kernel.empty() || (kernelRows % 2 == 0))
    {
        sys_error("the kernel you passed is invalid");
    }
    if (kernelRows == 1 && kernelCols == 1)
    {
        sys_error("kernel is invalid, you should ensure the size kernel is (m, m), or (1, m) or (m, 1)...");
    }
    // int halfKernelRows = (kernelRows / 2 == 0) ? 1 : (kernelRows / 2);
    // int halfKernelCols = (kernelCols / 2 == 0) ? 1 : (kernelCols / 2);
    int halfKernelRows = kernelRows / 2;
    int halfKernelCols = kernelCols / 2;
    // first, you should define a new Mat, you should use zero padding.
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    Mat tempMat = Mat::zeros(Size(cols + kernelCols - 1, rows + kernelRows - 1), CV_64F);
    Mat tempMat_ = tempMat(Rect(halfKernelCols, halfKernelRows, cols, rows));
    Mat inputImageMat;
    inputImage.convertTo(inputImageMat, CV_64F);
    // you should ensure the similar data type in inputImageMat and tempMat_.
    inputImageMat.copyTo(tempMat_);

    double minValue, maxValue;
    minMaxLoc(tempMat, &minValue, &maxValue, 0, 0);
    kernel.convertTo(kernel, CV_64F);
    Mat tempMat__ = Mat::zeros(tempMat.size(), CV_64F);
    Mat scannedMat;
    double *scannedMatRow;
    double *kernelRowMat;
    for (int y = halfKernelRows; y < rows; y++)
    {
        for (int x = halfKernelCols; x < cols; x++)
        {
            // notice, here we do not use the deep copy, it means the data in scannedMat is related to the tempMat.
            // so we can change the scannedMat directly, the tempMat will also be changed.
            // you should add .clone() when you created the scannedMat if you want to use deep copy.
            scannedMat = tempMat(cv::Rect(x - halfKernelCols, y - halfKernelRows, kernelCols, kernelRows));
            double sumValue = 0.0;
            for (int i = 0; i < scannedMat.rows; i++)
            {
                scannedMatRow = scannedMat.ptr<double>(i);
                kernelRowMat = kernel.ptr<double>(i);
                for (int j = 0; j < scannedMat.cols; j++)
                {
                    sumValue += (scannedMatRow[j] * kernelRowMat[j]);
                }
            }
            tempMat__.at<double>(y, x) = sumValue;
        }
    }
    // notice sum function will return a Scalar what is the class defined in opencv.
    // it is a one dimension array, you should index it and receive it used double if you want to sum one 
    // single channel Mat.
    double sumMat = cv::sum(kernel)[0];
    if (sumMat == 0)
    {
        Mat addImage, tempMat___;
        // laplacian, we can inference you want to sharpen the image.
        // you can not use the linearScaling method to mapping the gray value to range(0, 255)
        // you should use the method saturate_cast, gray value = (gray value > 255, < 0) : (255, 0) ? (gray value)
        cv::add(tempMat, tempMat__, addImage);
        double *addImageRow, *tempMatRow, *tempMat__Row;
        double minValue, maxValue;
        cv::minMaxLoc(kernel, &minValue, &maxValue, 0, 0);
        int c = (maxValue > 1) ? 1 : -1;
        for (int i = 0; i < addImage.rows; i++)
        {
            addImageRow = addImage.ptr<double>(i);
            tempMatRow = tempMat.ptr<double>(i);
            tempMat__Row = tempMat__.ptr<double>(i);
            for (int j = 0; j < addImage.cols; j++)
            {
                addImageRow[j] = cv::saturate_cast<double>(tempMatRow[j] + c * tempMat__Row[j]);
            }
        }
        addImage.convertTo(outputImage, CV_8UC1);
        return;
    }
    linearScaling(tempMat__, outputImage);
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-21 15:08:44
 * @Parameters: 
 * @Return: 
 * @Description: spatial convolution operation, you should distingush the different between
 * spatial filter and spatial convolution. the spatial filter is also named as spatial correlation.
 * we have define it in the former function spatialFilterOperation. then, we will consider the 
 * spatial convolution. the convolution will rotation the kernel 180 degrees based on the orginal kernel.
 * and the spatial filter used the original kernel, so it is the difference between them. note, rotated 180
 * degrees is not the transpose, transpose is 90 degrees and has the order. rotation is different from it.
 * we will define a function to rotate one matrix. notice, it is similar to the spatial correlation, 
 * you should ensure the size of kernel is an odd number. and you should do the zero padding.
 * do four edges zero padding and each edge padding (size-1)/2 numbers rows or columns zero.
 * 
 * then, we will define a generally used zero padding function and a matrix rotation function.
 * and you should consider some special word for these operation. just like we have defined
 * the spatial correlation and spatial convolution. then we will define the different words
 * about the non zero padding and zero padding. the result of the kernel operation will be named
 * correlation result if you have not defined the zero padding, it will be named as complated correlation
 * result if you have done the zero padding. of course, the spatial convolution also has these two definition.
 * the convolution result and complete convolution result.
 * 
 * 
 * 
 */
void spatialConvolution(Mat &inputImage, Mat &outputImage, Mat kernel) 
{
    rotationMat(kernel, ONEEIGHTZERO);
    spatialFilterOperation(inputImage, outputImage, kernel);
}


/**
 * @Author: weiyutao
 * @Date: 2023-02-21 18:04:55
 * @Parameters: 
 * @Return: 
 * @Description: then, we can define a generally function what can select the filter or kernel.
 * it means you can select one method. filter or kernel, the difference between them is 
 * the kernel will rotate 180 degrees based on the filter. we named the function as spatialFilter, 
 * it involved the spatial filter and spatial convolution. it can also be named as 
 * correlation and convolution. we will set two macro CORRELATION AND CONVOLUTION to provide 
 * the selected. default is coorelation. we have not found the essence difference between the correlation
 * and convolution so far. probably because the feature exchange law of convlotion result to it will be more
 * simple to do multiply kernels convolution operation. we will consider the deep differnce between them.
 * put the feature aside, the result of correlation is similar to the result of convolution.
 */
void spatialFilter(Mat &inputImage, Mat &outputImage, Mat kernel, int model = CORRELATION) 
{
    if (model == CONVOLUTION)
    {
        spatialConvolution(inputImage, outputImage, kernel);
    }
    spatialFilterOperation(inputImage, outputImage, kernel);
}

void spatialFilterUsedSeparatedKernel(Mat &inputImage, Mat &outputImage, Mat kernel, int model = CORRELATION) 
{
    // if you have used the original operation that correlation, you can not use the 
    // separated kernel because the correlation operation is not suitable for the 
    // exchange law and associative law.
    if (model == CONVOLUTION)
    {
        // you should redefine the spatial convolution function that add the function operated
        // based on 1 dimension kernel.
        // you should judge the rank of the kernel.
        int rank = getRankFromMat(kernel);
        cout << rank << endl;
        if (rank != 1)
        {
            spatialConvolution(inputImage, outputImage, kernel);
            return;
        }
        Mat w1, w2;
        separateKernel(kernel, w1, w2);
        spatialConvolution(inputImage, outputImage, w1);
        spatialConvolution(outputImage, outputImage, w2);
        return;
    }
    spatialFilterOperation(inputImage, outputImage, kernel);
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-23 15:48:03
 * @Parameters: kernelSize, one odd number.
 * @Return: 
 * @Description: 
 */
void medianFilter(Mat &inputImage, Mat &outputImage, int kernelSize) 
{
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int halfKernel = kernelSize >> 1;
    outputImage = Mat::zeros(cols + halfKernel, rows + halfKernel, CV_8UC1);
    Mat tempImage = outputImage(Rect(halfKernel, halfKernel, cols, rows));
    Mat tempKernelImage, tempSortKernelImage;
    inputImage.copyTo(tempImage);
    for (int i = halfKernel; i < rows; i++)
    {
        for (int j = halfKernel; j < cols; j++)
        {
            tempKernelImage = outputImage(Rect((j - halfKernel), (i - halfKernel), kernelSize, kernelSize)).clone();
            tempKernelImage = tempKernelImage.reshape(1, 1);
            cv::sort(tempKernelImage, tempSortKernelImage, 0);
            swap(outputImage.at<uchar>(i, j), tempSortKernelImage.at<uchar>(0, (kernelSize * kernelSize / 2)));
        }
    }
}