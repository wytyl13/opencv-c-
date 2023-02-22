/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-02-22 16:44:42
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-02-22 16:44:42
 * @Description: this file will introduct the frequency domain filter.
 * the frequency is one important domain in exchange domain.
 * the link between spatial domain and frequency domain is fourier transform.
 * 
 * started from here, we will operate all the transformation based on the
 * exchange domain, just like frequecy domain, we will enter to the
 * frequency domain from spatial domain and do some transformation
 * in the frequency domain, and then return to the spatial domain. 
 * 
 * the difference between spatial filter and exchange filter is
 * the convolution is the basic for the spatial filter.
 * it is equal to the product in frequency.
 * the impulse of the amplitude value is A in spatial domain is euqal to the
 * constant that value is A in frequency domain.
 * 
***********************************************************************/


#include "../include/frequencyFilter.h"