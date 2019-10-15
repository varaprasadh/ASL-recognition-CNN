import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)

    lower_red = np.array([0,0,0])
    upper_red = np.array([255,255,255])

    mask = cv2.inRange(hsv ,lower_red ,upper_red)
    res = cv2.bitwise_and(frame, frame ,mask =mask)

    kernel = np.ones((15,15),np.float32)/225
    smooth = cv2.filter2D(res ,-1,kernel)

    gaussianBlur = cv2.GaussianBlur(res,(15,15),0)
    medianBlur = cv2.medianBlur(res,15)
    bilateralBlur = cv2.bilateralFilter(res ,15,75,75)

    cv2.imshow('mask',mask)
    cv2.imshow('Averaging',smooth)
    cv2.imshow('Gaussian',gaussianBlur)
    cv2.imshow('Median',medianBlur)
    cv2.imshow('Bilateral',bilateralBlur)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
