"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

# def gauss1d(x, mean, sigma):
#     '''
#     1d gaussian with mean and sigma
#     :param x:
#     :param mean:
#     :param sigma:
#     :return fx:
#     '''
#     dist = (x-mean)**2
#     stand_dist = dist/(2*sigma**2)
#     fx = np.exp(-stand_dist)/(np.sqrt(2*np.pi)*sigma)
#     return fx

###
#please coding here for solving Task [I].

file_name = './test.jpg'
img = cv2.imread(file_name)
cv2.imshow('origin',img)
redI = np.zeros_like(img)
redI[:,:,2]   = img[:,:,2]
cv2.imshow('red', redI)
beans =[256]
r_hist = cv2.calcHist([img],[2],None,beans,[0,255])
plt.plot(r_hist,color = 'r')
plt.xlim([0,256])
plt.show()
cv2.destroyWindow('red')

###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

###
#please coding here for solving Task[II]

from scipy import ndimage
for i in range(8):
    gaussian_pic = ndimage.gaussian_filter(img, sigma=i)#高斯滤波创建8种
    cv2.imshow('gaussian sigema = '+str(i), gaussian_pic)
cv2.waitKey()
cv2.destroyAllWindows()







"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]

mean = (1,1)
cov = np.eye(2)
x, y = np.random.multivariate_normal(mean, cov, 50000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
plt.hist(x.ravel(), bins=256, color='b')#这是x方向上点的分布情况
plt.show()
plt.hist(y.ravel(), bins=256, color='b')#这是x方向上点的分布情况
plt.show()


cv2.waitKey()

