/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-03-01 16:26:12
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-03-01 16:26:12
 * @Description: why base64? transform the binary data to ascii, then transform
 * ascii to base64, you can show all informations what you want to show
 * used base64 coding. because base64 just has 64 character, A-Z, a-z and 0-9, / + =(26+26+10+2=64), 
 * = is the suffx. from 0 to 63, so you can use
 * 6 bits to show it. just like 00 0000(0) - 11 1111(2^6-1=63), but notice, the base64 coding
 * need to 4/3 times size of binary data. so it will increase the data size you want to transmission.
 * but it is stability and supported the json and http request transmission. so it is generally used in
 * the B/S model. then we will implement the codeBase64 function. it is dedicated to the transmissioning
 * about the image. of course, you can also use it in the application of other text file. of course, you can
 * also use it to encode the url or the request content.
 */
#ifndef _BASE64H_
#define _BASE64H_
#include "general.h"
class Base64
{
public:
	static std::string base64Decode(const unsigned char *Data, int DataByte)
	{
		//解码表
		const unsigned char DecodeTable[] =
		{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			62, // '+'
			0, 0, 0,
			63, // '/'
			52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
			0, 0, 0, 0, 0, 0, 0,
			0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
			0, 0, 0, 0, 0, 0,
			26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
			39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
		};
		std::string strDecode;
		int nValue;
		int i = 0;
		while (i < DataByte) {
			if (*Data != '\r' && *Data != '\n') {
				nValue = DecodeTable[*Data++] << 18;
				nValue += DecodeTable[*Data++] << 12;
				strDecode += (nValue & 0x00FF0000) >> 16;
				if (*Data != '=') {
					nValue += DecodeTable[*Data++] << 6;
					strDecode += (nValue & 0x0000FF00) >> 8;
					if (*Data != '=') {
						nValue += DecodeTable[*Data++];
						strDecode += nValue & 0x000000FF;
					}
				}
				i += 4;
			}
			else {
				Data++;
				i++;
			}
		}
		return strDecode;
	}

	static std::string base64Encode(const unsigned char* Data, int DataByte) {
		//编码表
		const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		//返回值
		std::string strEncode;
		unsigned char Tmp[4] = { 0 };
		int LineLength = 0;
		for (int i = 0; i < (int)(DataByte / 3); i++) {
			Tmp[1] = *Data++;
			Tmp[2] = *Data++;
			Tmp[3] = *Data++;
			strEncode += EncodeTable[Tmp[1] >> 2];
			strEncode += EncodeTable[((Tmp[1] << 4) | (Tmp[2] >> 4)) & 0x3F];
			strEncode += EncodeTable[((Tmp[2] << 2) | (Tmp[3] >> 6)) & 0x3F];
			strEncode += EncodeTable[Tmp[3] & 0x3F];
			if (LineLength += 4, LineLength == 76) { strEncode += "\r\n"; LineLength = 0; }
		}
		//对剩余数据进行编码
		int Mod = DataByte % 3;
		if (Mod == 1) {
			Tmp[1] = *Data++;
			strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
			strEncode += EncodeTable[((Tmp[1] & 0x03) << 4)];
			strEncode += "==";
		}
		else if (Mod == 2) {
			Tmp[1] = *Data++;
			Tmp[2] = *Data++;
			strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
			strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
			strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2)];
			strEncode += "=";
		}


		return strEncode;
	}
	

	static std::string Mat2Base64(const cv::Mat &img, std::string imgType)
	{
		//Mat转base64
		std::string img_data;
		std::vector<uchar> vecImg;
		std::vector<int> vecCompression_params;
		vecCompression_params.push_back(IMWRITE_JPEG_QUALITY);
		vecCompression_params.push_back(90);
		imgType = "." + imgType;
		cv::imencode(imgType, img, vecImg, vecCompression_params);
		img_data = base64Encode(vecImg.data(), vecImg.size());
		return img_data;
	}
	
	static cv::Mat Base2Mat(std::string &base64_data)
	{
		cv::Mat img;
		std::string s_mat;
		s_mat = base64Decode((const unsigned char *)base64_data.data(), base64_data.size());
		std::vector<char> base64_img(s_mat.begin(), s_mat.end());
		img = cv::imdecode(base64_img, IMREAD_COLOR);
		return img;
	}
};

#endif