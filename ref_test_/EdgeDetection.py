import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    _,frame = cap.read()
    usv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,0,0])
    upper_red = np.array([255,255,255])

    mask = cv2.inRange(usv , lower_red , upper_red)
    res = cv2.bitwise_and(frame, frame , mask = mask)

    laplace = cv2.Laplacian(frame ,cv2.CV_64F)
    sobelX = cv2.Sobel(frame,cv2.CV_64F ,1,0,ksize =5)
    sobelY = cv2.Sobel(frame,cv2.CV_64F ,0,1,ksize =5)

    cv2.imshow('Original',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('laplacian',laplace)
    cv2.imshow('sobelx',sobelX)
    cv2.imshow('sobely',sobelY)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
