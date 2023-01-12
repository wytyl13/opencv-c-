import cv2
import numpy as np

path = "../resources/women.png"
image = cv2.imread(path)

height, width = image.shape[:2]
# this int(width * 0.5), int(height * 0.5)) corresponding to the Size in c++
# the first param is width, the second param is height, you should pass int.
imageResize = cv2.resize(image, (int(width * 0.5), int(height * 0.5)), interpolation = cv2.INTER_CUBIC)
imageGray = cv2.cvtColor(imageResize, cv2.COLOR_BGR2GRAY)
cv2.imshow("test", imageResize)
cv2.imshow("test", imageGray)
cv2.waitKey(0)

test = cv2.Mat.eyes(3, 3, CV_8UC1)
print(test)

# print(height, width)




# print(image2)
# # imageGray = cv2.CreateMat(h, w, CV2_32FC3)
# # cv2.resize(image, imageGray, 360, 600);
# # imageGray = cv2.zeros(cv2.Size(image), CV_8UC1);
# # cv2.cvtColor(image, image3, COLOR_BGR2GRAY);
# print(image)
# cv2.imshow("the original image", image);
# # cv2.imshow("the original image", resizeImage);
# cv2.waitKey(0);



