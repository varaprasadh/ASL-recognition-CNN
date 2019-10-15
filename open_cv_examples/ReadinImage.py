import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)

#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(img ,cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.plot([30,100], [80,100] , 'c', linewidth = 5)
plt.show()
