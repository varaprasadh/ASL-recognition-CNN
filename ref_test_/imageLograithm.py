import cv2
import numpy as np

img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainlogo.png')

rows, cols,channel = img2.shape
roi = img1[0:rows, 0:cols]

img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
img2_bg = cv2.bitwise_and(img2,img2,mask=mask)

#cv2.imshow('img12_bg',img2_bg)
cv2.imshow('img1_bg',img1_bg)
cv2.imshow('maskInverse',mask_inv)
cv2.imshow('mask',mask)
cv2.imshow('roi',roi)
#cv2.imshow('Colored',img2)
#cv2.imshow('GrayImage',img2gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
