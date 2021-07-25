import cv2 as cv
import numpy as np 
import time
import mediapipe as mp
cap=cv.VideoCapture(0)
hands=mp.solutions.hands
hands_mesh=hands.Hands()
mpDraw=mp.solutions.drawing_utils
pTime=0

cTime=0


while  True:
    _,frm=cap.read()
    rgb=cv.cvtColor(frm,cv.COLOR_BGR2RGB)
    op=hands_mesh.process(rgb)
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            for id, lm in enumerate(i.landmark):
                print(id,lm)
                h,w,c=frm.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==8:
                    cv.circle(frm,(cx,cy),15,(255,0,255),cv.FILLED)
            mpDraw.draw_landmarks(frm,i,hands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv.putText(frm,str(int(fps)),(10,70),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,3,(255,0,255),3)
    image=cv.flip(frm,1)
    cv.imshow('windows',image)

    if cv.waitKey(1)==27:
        cv.destroyAllWindows()
        cap.release()
        break