import cv2
import numpy as np

img = cv2.imread('New Doc 2017-09-06_2.jpg')
resize_img = cv2.resize(img,(50,50))
cv2.imshow('New image',resize_img)
cv2.imwrite('aadhar.jpg',resize_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
