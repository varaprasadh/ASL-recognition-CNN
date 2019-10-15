import cv2
import numpy as np

img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
img[100,100] = [255,255,255]

img[100:200,100:200] = [255,255,4]
watch_face = img[37:111,107:194]
img[0:74,0:87]= watch_face

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
