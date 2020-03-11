import cv2
import numpy as np
import time
# import copy

#create face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

#Open Camera object
cap = cv2.VideoCapture(0)


def Angle(v1, v2):
 dot = np.dot(v1, v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists

def FindDistance(A, B):
 return np.sqrt(np.power((A[0][0]-B[0][0]), 2) + np.power((A[0][1]-B[0][1]), 2))


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

while(1):
    try:

    #Measure execution time
        start_time = time.time()

        #Capture frames from the camera
        ret, frame = cap.read()
        # original_frame=copy.deepcopy(frame)
        #convert to grayscale mode 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        #apply face haarcascade classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            cv2.imshow('face_gray', roi_gray)
            cv2.imshow('face_color', roi_color)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), -1)
            
        #Blur the image
        blur = cv2.blur(frame, (3, 3))

        #Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv, np.array([2, 50, 50]), np.array([15, 255, 255]))

        #Kernel matrices for morphological transformation
        kernel_square = np.ones((11, 11), np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        #Perform morphological transformations to filter out the background noise
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

        #Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        max_area = 100
        ci = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                ci = i

        #Largest area contour
        cnts = contours[ci]

        #Find convex hull
        hull = cv2.convexHull(cnts)

        #Find convex defects
        hull2 = cv2.convexHull(cnts, returnPoints=False)
        defects = cv2.convexityDefects(cnts, hull2)

        #Get defect points and draw them in the original image
        FarDefect = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame, start, end, [0, 255, 0], 1)

        #Find moments of the largest contour
        moments = cv2.moments(cnts)

        #Central mass of first order moments
        if moments['m00'] != 0:
            cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
            cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
        centerMass = (cx, cy)

        #Draw center mass
        cv2.circle(frame, centerMass, 7, [100, 0, 255], 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        x, y, w, h = cv2.boundingRect(cnts)
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.drawContours(frame, [hull], -1, (255, 255, 255), 2)
        
        roi_hand = frame[y:y+h, x:x+w]
        padded=getPaddedImage(roi_hand)
        cv2.imshow("square",padded)
        # roi_hand_gray=cv2.cvtColor(roi_hand,cv2.COLOR_BGR2GRAY)
        # roi_hand_thresh=cv2.threshold(roi_gray,150,255,cv2.THRESH_BINARY)
        cv2.imshow('run time frame', frame)
        cv2.imshow('run time hand', roi_hand)
    except:
        continue
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
