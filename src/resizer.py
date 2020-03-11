#Importing modules opencv + numpy
import cv2
import numpy as np

def getPaddedImage(img):

    #Getting the bigger side of the image
    s = max(img.shape[0:2])

    #Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    #Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    return f


f = getPaddedImage(cv2.imread('source1.jpg'))
#Showing results (just in case)
cv2.imshow("IMG", f)

#A pause, waiting for any press in keyboard
cv2.waitKey(0)

#Saving the image
cv2.imwrite("img2square.png",f)
cv2.destroyAllWindows()
