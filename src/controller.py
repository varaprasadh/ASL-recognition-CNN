import cv2
import time

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# import copy

labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12,
               'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
               'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}
    
_labels = dict([(value, key) for key, value in labels_dict.items()])

img_width, img_height = 64, 64

model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Open Camera object
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
counter=0

def getPaddedImage(img):

    # Getting the bigger side of the image
    s = max(img.shape[0:2])

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    return f
 

while(1):
    try:

        # Measure execution time
        start_time = time.time()

        # Capture frames from the camera
        ret, frame = cap.read()
        # original_frame=copy.deepcopy(frame)

        # convert to grayscale mode
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur the image
        blur = cv2.blur(frame, (3, 3))

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array(
            [2, 50, 50]), np.array([15, 255, 255]))

        # Kernel matrices for morphological transformation
        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Perform morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
        erosion = cv2.erode(dilation, kernel_square, iterations=1)
        dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
        filtered = cv2.medianBlur(dilation2, 5)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
        median = cv2.medianBlur(dilation2, 5)
        ret, thresh = cv2.threshold(median, 127, 255, 0)

        cv2.rectangle(frame, (0, 0), (200, 200), (255, 255, 255))
        roi=frame[0:200,0:200]


        #img = image.load_img('./test_ext/sample_1_l.jpg',
        #                     target_size=(img_width, img_height))
        img = cv2.resize(roi, (img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10)
        print("predictions : ",classes)
        print("predicted label : ",_labels.get(classes[0]))
        cv2.putText(frame,_labels.get(classes[0]),(300,300),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow("cropped",roi)

        cv2.imshow('main window', frame)
    except:
        break
    k = cv2.waitKey(5) & 0xFF
    #debug purpose
    if k==ord(' '):
        cv2.imwrite('./test_generated/'+'img_'+str(counter)+'.png', img)
        counter+=1
    
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
