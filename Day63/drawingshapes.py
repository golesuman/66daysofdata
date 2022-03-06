import cv2
import  numpy as np
img=np.zeros((512,512,3),np.uint8)

lineimage=cv2.line(img,(0,0),(100,500),(0,255,0),3)
rectangle=cv2.rectangle(img,(0,0),(250,250),(255,255,0),4)
cv2.putText(img,"Opencv",(300,100),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=3,color=(255,0,0))
cv2.imshow("line",lineimage)
cv2.waitKey(0)