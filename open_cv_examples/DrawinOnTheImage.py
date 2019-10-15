import cv2
import numpy as np

img = cv2.imread('DP.jpg' , cv2.IMREAD_COLOR)

cv2.line(img, (0,0) ,(150,150), (255,255,255), 15)
cv2.rectangle(img , (120,25) , (300,90), (0,255,0) ,10)
cv2.circle(img , (50,50), 20, (0,0,255), -1)

pts = np.array([[10,5],[15,25],[25,50],[40,60],[60,50]],np.int32)
cv2.polylines(img, [pts], False, (0,255,255), 5)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'UTKARSH', (0,100), font ,1,(200,255,69),5,cv2.LINE_AA)

cv2.imshow('Utkarsh',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

