#!/usr/bin/python
#encoding:utf-8

"""
Develop Date: 18/06/2019
Develop Subject: My Project
Developer: Matt Wang
"""

import sys
import os
from PyQt4 import QtCore,QtGui,uic
import numpy as np
import cv2
import os
import dlib
import imutils
reload(sys).setdefaultencoding("utf8")

Ui_MainWindow,QtBaseClass=uic.loadUiType("test.ui")

fname=""
n=0

class Myapp(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
# --------------GUI--------------------------------------------------------
        self.pushButton.clicked.connect(self.takePicture)
        self.pushButton_2.clicked.connect(self.faceDetection)
        self.pushButton_3.clicked.connect(self.fiveSensesDetection)
        self.pushButton_4.clicked.connect(self.nextPicture)
        self.pushButton_5.clicked.connect(self.previousPicture)
        self.pushButton_6.clicked.connect(self.fun6) # waiting for change rgb
        self.pushButton_8.clicked.connect(self.openFile)
        self.pushButton_7.clicked.connect(self.recodeVideo)

        self.dial.valueChanged.connect(self.lcdNumber.display)
        self.dial_2.valueChanged.connect(self.lcdNumber_2.display)
        self.dial_3.valueChanged.connect(self.lcdNumber_3.display)

# ---------------------------------------------------------------------
    def takePicture(self): #pushButton
        # please add recode function below
        # and shot picture function
        cap = cv2.VideoCapture(0)

        if not os.path.exists("myPicture"):
            os.mkdir("myPicture")
        mypath = "./myPicture/"
        files = os.listdir(mypath)
        fn=len(files)

        p=1

        while(cap.isOpened()):
            ret , frame = cap.read()
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow("Press 'z' to take a picture or press 'q' to exit",frame)
            k=cv2.waitKey(1)
            if k==ord("z") or k==ord("Z"): # take a picture
                if fn==0:
                    cv2.imwrite("myPicture\\pic"+str(p)+".jpg",frame)
                    p+=1
                else:
                    fn+=1
                    cv2.imwrite("myPicture\\pic"+str(fn)+".jpg",frame)
            if cv2.waitKey(1) & 0xFF ==ord('q'): # quit
                break
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------
    def faceDetection(self):

        cap=cv2.VideoCapture(0)
        detector=dlib.get_frontal_face_detector()

        p=1

        if not os.path.exists("faceDetection"):
            os.mkdir("faceDetection")
        mypath = "./faceDetection/"
        files = os.listdir(mypath)
        fn=len(files)

        while(cap.isOpened()):
            try:
                ret , frame = cap.read()
                # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                face_rects=detector(frame,0)
                for i, d in enumerate(face_rects):
                    x1=d.left()
                    y1=d.top()
                    x2=d.right()
                    y2=d.bottom()

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                if ret==True:
                    cv2.imshow("Press 'z' to take a picture of face or press 'q' to exit",frame)
                    k=cv2.waitKey(1)
                    if k==ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    if k==ord("z") or k==ord("Z"): # press z to take a picture
                        if fn==0:
                            crop_frame = frame[y1:y2, x1:x2]
                            cv2.imwrite("faceDetection\\face"+str(p)+".jpg",crop_frame)
                            p=p+1
                        else:
                            fn+=1
                            crop_frame = frame[y1:y2, x1:x2]
                            cv2.imwrite("faceDetection\\face"+str(fn)+".jpg",crop_frame)
                else:
                    cap.release()
                    cv2.destroyAllWindows()
                    break

            except:
                cap.release()
                cv2.destroyAllWindows()
                break
        cap.release()
        cv2.destroyAllWindows()

# ----------------------------------------------------------------
    def fiveSensesDetection(self):

        cap=cv2.VideoCapture(0)
        detector=dlib.get_frontal_face_detector()
        predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        while(cap.isOpened()):
            ret , frame = cap.read()
            # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_rects,scores,idx=detector.run(frame,0)
            for i, d in enumerate(face_rects):
                x1=d.left()-20
                y1=d.top()-30
                x2=d.right()+20
                y2=d.bottom()+20
                t="%2.2f(%d)"%(scores[i],idx[i])

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4) # 人臉辨識方框
                # 給人臉打分數, 分數越高越有可能是真的人臉
                cv2.putText(frame,t,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255),1)

                # 特徵辨識的n個圓點 ( 1-20: 臉框 , 20-68: 五官 )

                lanf=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                shape=predictor(lanf,d)

                for i in range(68):
                    cv2.circle(frame,(shape.part(i).x,shape.part(i).y),3,(0,0,255),2)
                    cv2.putText(frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

            cv2.imshow("frame",frame)
            if cv2.waitKey(1)==ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
# ----------------------------------------------------------------
    def nextPicture(self):

        global fname
        global n

        a=fname.split("/")[-1] # picture name
        nfile=fname.split("/")[-2] # folder name
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath)
        nf=len(files) # number of file in that folder

        imgPath="./"+nfile+"/"+files[n]

        n+=1
        print(n)

        if n==nf+1:
            n=0
            imgPath="./"+nfile+"/"+files[n]
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
            self.lineEdit.setText(imgPath)
            print("111111",n)
        else:

            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
            self.lineEdit.setText(imgPath)
            print("2-->",n)

        print(files)



        # global count #1

        # mypath = "./myPicture/"
        # files = os.listdir(mypath)
        # fileslen=len(files)

        # imgPath="./myPicture/pic"+str(count)+".jpg"
        # text="You are watching: No.[%d] picture "%count+"(pic%d.jpg)"%count
        # count+=1

        # if count == fileslen+1:
        #     count=1
        #     self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(320,320,QtCore.Qt.KeepAspectRatio))
        #     self.label_2.setText(text)
        # else:
        #     self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(320,320,QtCore.Qt.KeepAspectRatio))
        #     self.label_2.setText(text)
# ----------------------------------------------------------------
    def previousPicture(self):

        global count #1

        mypath = "./myPicture/"
        files = os.listdir(mypath)
        fileslen=len(files)

        imgPath="./myPicture/pic"+str(count)+".jpg"
        text="You are watching: No.[%d] picture "%count+"(pic%d.jpg)"%count
        count-=1

        if count == 0:
            count=fileslen
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(320,320,QtCore.Qt.KeepAspectRatio))
            self.label_2.setText(text)
        else:
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(320,320,QtCore.Qt.KeepAspectRatio))
            self.label_2.setText(text)
# ----------------------------------------------------------------
    def fun6(self):

        # global count #1

        # imgPath="./myPicture/pic"+str(count)+".jpg"
        # self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(320,320,QtCore.Qt.KeepAspectRatio))

        # v=self.dial.value() # get dial value
        img=cv2.imread("a.jpg")
        total=img.shape

        for i in range(0,total[0]):
            for j in range(0,total[1]):
                # img[i,j,0]=img[i,j,0]/3 #R
                # img[i,j,1]=img[i,j,1]/3 #G
                # img[i,j,2]=img[i,j,2]/3 #B
                img[i,j,0]=self.dial.value()
                img[i,j,1]=self.dial_2.value()
                img[i,j,2]=self.dial_3.value()
        print(img[i,j,0])
        print(img[i,j,1])
        print(img[i,j,2])

        while(1):
            cv2.imshow("image",img)

            k=cv2.waitKey(1)
            if k==ord('q'):
                break
        cv2.destroyAllWindows()
# ----------------------------------------------------------------
    def openFile(self):

        global fname
        global n

        path="D:\project\myPicture"
        fname=QtGui.QFileDialog.getOpenFileName(self,"Open file",path,"Image files (*.jpg *.gif)")
        self.label.setPixmap(QtGui.QPixmap(fname))
        self.lineEdit.setText(fname)

        a=fname.split("/")[-1] # picture name
        nfile=fname.split("/")[-2] # folder name
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath)
        n=files.index(a) # number of picture in list
# ----------------------------------------------------------------
    def recodeVideo(self):
        cap = cv2.VideoCapture(0)

        fourcc = cv2.cv.CV_FOURCC("D","I","B"," ")

        mypath = "./myVideo/"
        files = os.listdir(mypath)
        fileslen=len(files)
        fileslen+=1

        out = cv2.VideoWriter("myVideo\\video"+str(fileslen)+".avi", fourcc,30, (640, 480))

        if not os.path.exists("myVideo"): # if this folder doesn't exists
            os.mkdir("myVideo")

        while(cap.isOpened()):
            ret,frame = cap.read()
            if ret == True:

                out.write(frame)

                cv2.imshow("Recoding...", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    app=QtGui.QApplication(sys.argv)
    w=Myapp()
    w.show()
    sys.exit(app.exec_())