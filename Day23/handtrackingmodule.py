import cv2 as cv
import numpy as np 
import time
import mediapipe as mp

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,tractCon=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.tractCon=tractCon

        self.hands=mp.solutions.hands
        self.hands_mesh=self.hands.Hands(self.mode,self.maxHands,self.detectionCon,self.tractCon)
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,frm,draw=True):
        rgb=cv.cvtColor(frm,cv.COLOR_BGR2RGB)
        self.op=self.hands_mesh.process(rgb)
        if self.op.multi_hand_landmarks:
            for i in self.op.multi_hand_landmarks:
                 self.mpDraw.draw_landmarks(frm,i,self.hands.HAND_CONNECTIONS)
        return frm


    def findPosition(self,frm,handNo=0,draw=True):
        lmList=[]
        if self.op.multi_hand_landmarks:
            myHand=self.op.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c=frm.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(frm,(cx,cy),15,(255,0,255),cv.FILLED)
        return lmList


def main():
    pTime=0

    cTime=0

    cap=cv.VideoCapture(0)
    detector=handDetector()
    while  True:
        _,frm=cap.read()
        frm=detector.findHands(frm)
        lmList=detector.findPosition(frm)
        if len(lmList)!=0:
            print(lmList[4])
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

if __name__ == '__main__':
    main()