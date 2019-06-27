#!/usr/bin/python3
#encoding:utf-8

"""
Develop Date: 18/06/2019 - 26/06/2019
Develop Subject: Python_Opencv_FaceDetection
Developer: Matt Wang
Python environment: version 3.6
"""

import sys
import os
from PyQt4 import QtCore,QtGui,uic
import numpy as np
import cv2
import dlib
import imutils
import datetime
import face_recognition
# reload(sys).setdefaultencoding("utf8")

Ui_MainWindow,QtBaseClass=uic.loadUiType("project1.ui")

class Myapp(QtGui.QMainWindow,Ui_MainWindow):
    strimg=""
    fname=""
    n=0
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
        self.pushButton_6.clicked.connect(self.saveImage)
        self.pushButton_8.clicked.connect(self.openFile)
        self.pushButton_7.clicked.connect(self.recodeVideo)
        self.pushButton_10.clicked.connect(self.firstPic)
        self.pushButton_11.clicked.connect(self.LastPic)
        self.pushButton_9.clicked.connect(self.collectFace)
        self.pushButton_12.clicked.connect(self.facerecog)

        self.dial.valueChanged.connect(self.controlrgb)
        self.dial_2.valueChanged.connect(self.controlrgb)
        self.dial_3.valueChanged.connect(self.controlrgb)

        self.horizontalSlider.valueChanged.connect(self.bright)
        self.horizontalSlider_2.valueChanged.connect(self.blur)

# ---------------------------------------------------------------------
    def takePicture(self):

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
            if k==ord("q"):
                cap.release()
                cv2.destroyAllWindows()
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

        p=1

        if not os.path.exists("faceDetection1"):
            os.mkdir("faceDetection1")
        mypath = "./faceDetection1/"
        files = os.listdir(mypath)
        fn=len(files)

        while(cap.isOpened()):
            try:
                ret , frame = cap.read()
                # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                face_rects,scores,idx=detector.run(frame,0)
                for i, d in enumerate(face_rects):
                    x1=d.left()
                    y1=d.top()
                    x2=d.right()
                    y2=d.bottom()
                    text="%2.2f(%d)"%(scores[i],idx[i])

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                    cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255),1)
                    lanf=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                    shape=predictor(lanf,d)

                    for i in range(68):
                        cv2.circle(frame,(shape.part(i).x,shape.part(i).y),3,(0,0,255),2)
                        cv2.putText(frame,str(i),(shape.part(i).x,shape.part(i).y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

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
                            cv2.imwrite("faceDetection1\\face"+str(p)+".jpg",crop_frame)
                            p=p+1
                        else:
                            fn+=1
                            crop_frame = frame[y1:y2, x1:x2]
                            cv2.imwrite("faceDetection1\\face"+str(fn)+".jpg",crop_frame)
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
    def nextPicture(self):

        global n
        global fname

        if not os.path.exists("temp"):
            os.mkdir("temp")

        nfile=fname.split("/")[-2] # folder name
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath)
        nf=len(files) # number of file in that folder

        n+=1

        if n==nf:
            n=0
            imgPath="./"+nfile+"/"+files[n]
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
            self.lineEdit.setText(imgPath)
            img=cv2.imread(imgPath)
            cv2.imwrite("./temp/change1.jpg",img)

        else:
            imgPath="./"+nfile+"/"+files[n]
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
            self.lineEdit.setText(imgPath)
            img=cv2.imread(imgPath)
            cv2.imwrite("./temp/change1.jpg",img)

# ----------------------------------------------------------------
    def previousPicture(self):

        global n
        global fname

        if not os.path.exists("temp"):
            os.mkdir("temp")

        nfile=fname.split("/")[-2] # folder name
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath)
        nf=len(files) # number of file in that folder

        n-=1

        if n==-1:
            n=nf-1
            imgPath="./"+nfile+"/"+files[n]
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
            self.lineEdit.setText(imgPath)
            img=cv2.imread(imgPath)
            cv2.imwrite("./temp/change1.jpg",img)


        else:
            imgPath="./"+nfile+"/"+files[n]
            self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
            self.lineEdit.setText(imgPath)
            img=cv2.imread(imgPath)
            cv2.imwrite("./temp/change1.jpg",img)


# ----------------------------------------------------------------
    def bright(self): # path fixed!!

        n=self.horizontalSlider.value()
        n=n*50

        s=fname.replace("/","\\")
        img=cv2.imread(str(s))

        a=1.3

        x,y,chan=img.shape
        blank=np.zeros([x,y,chan],img.dtype)
        merge=cv2.addWeighted(img,a,blank,1-a,n)

        if not os.path.exists("temp"):
            os.mkdir("temp")

        cv2.imwrite("./temp/change1.jpg",merge)
        self.label.setPixmap(QtGui.QPixmap("./temp/change1.jpg"))

# ----------------------------------------------------------------
    def openFile(self):

        if not os.path.exists("temp"):
            os.mkdir("temp")

        global n
        global fname

        fname=QtGui.QFileDialog.getOpenFileName(self,"Open file","./myPicture/","Image files (*.jpg *.gif)")
        self.strimg=fname
        self.label.setPixmap(QtGui.QPixmap(fname))
        self.lineEdit.setText(fname)
        s=self.strimg
        img=cv2.imread(str(s))
        cv2.imwrite("./temp/change1.jpg",img)
        self.label.setPixmap(QtGui.QPixmap("./temp/change1.jpg"))

        a=fname.split("/")[-1] # picture name
        nfile=fname.split("/")[-2] # folder name
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath) #make a list and all file name within
        n=files.index(a) # number of picture in list
# ----------------------------------------------------------------
    def recodeVideo(self):

        if not os.path.exists("myVideo"):
            os.mkdir("myVideo")

        cap = cv2.VideoCapture(0)

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        mypath = "./myVideo/"
        files = os.listdir(mypath)
        fileslen=len(files)
        fileslen+=1

        out = cv2.VideoWriter()
        out.open("myVideo\\video"+str(fileslen)+".mp4",fourcc,20,size,True)

        while(cap.isOpened()):
            ret,frame = cap.read()
            if ret == True:
                out.write(frame)

                cv2.imshow("Please press 'q' to exit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# ----------------------------------------------------------------
    def blur(self):
        cblur=self.horizontalSlider_2.value()
        if cblur==2:
            cblur=5
        if cblur==3:
            cblur=15
        if cblur==4:
            cblur=19

        if not os.path.exists("temp"):
            os.mkdir("temp")

        s=fname.replace("/","\\")
        img=cv2.imread(str(s))

        blur=cv2.GaussianBlur(img,(cblur,cblur),0)

        cv2.imwrite("./temp/change1.jpg",blur)
        self.label.setPixmap(QtGui.QPixmap("./temp/change1.jpg"))

# ----------------------------------------------------------------
    def controlrgb(self):

        r=self.dial.value()
        r=r*51
        self.lcdNumber.display(r)

        g=self.dial_2.value()
        g=g*51
        self.lcdNumber_2.display(g)

        b=self.dial_3.value()
        b=b*51
        self.lcdNumber_3.display(b)

        s=fname.replace("/","\\")
        img=cv2.imread(str(s))

        a=1.3

        x,y,chan=img.shape
        red=np.zeros([x,y,chan],img.dtype)
        for i in range(0,x):
            for j in range(0,y):
                red[i,j]=[b,g,r]

        merge=cv2.addWeighted(img,a,red,1-a,n)

        cv2.imwrite("./temp/change1.jpg",merge)
        self.label.setPixmap(QtGui.QPixmap("./temp/change1.jpg"))

# ----------------------------------------------------------------
    def saveImage(self):

        if not os.path.exists("changedPic"):
            os.mkdir("changedPic")

        img=cv2.imread("./temp/change1.jpg")

        files=os.listdir("./changedPic/")
        fn=len(files)
        imgpath= "D:\\project\\changedPic"
        if fn==0:
            cv2.imwrite(imgpath+"\\cpic1.jpg",img)
        else:
            fn+=1
            cv2.imwrite(imgpath+"\\cpic"+str(fn)+".jpg",img)
        msg1=QtGui.QMessageBox()
        msg1.setWindowTitle("Image Saved")
        msg1.setInformativeText("Your picture has been saved in the changedPic folder!!")
        msg1.exec_()
# ----------------------------------------------------------------
    def firstPic(self):

        global n
        global fname

        a=fname.split("/")[-1]
        nfile=fname.split("/")[-2]
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath)
        n=files.index(a)

        imgPath="./"+nfile+"/"+files[0]
        self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
        self.lineEdit.setText(imgPath)
        img=cv2.imread(imgPath)
        cv2.imwrite("./temp/change1.jpg",img)

        n=0

# ----------------------------------------------------------------
    def LastPic(self):

        global n
        global fname

        a=fname.split("/")[-1]
        nfile=fname.split("/")[-2] # folder name
        mypath = "./"+nfile+"/"
        files = os.listdir(mypath)
        n=files.index(a)
        nf=len(files)
        nf=nf-1

        imgPath="./"+nfile+"/"+files[nf]
        self.label.setPixmap(QtGui.QPixmap(imgPath).scaled(640,480,QtCore.Qt.KeepAspectRatio))
        self.lineEdit.setText(imgPath)
        img=cv2.imread(imgPath)
        cv2.imwrite("./temp/change1.jpg",img)

        nf=len(files)
        n=nf-1



# ----------------------------------------------------------------
    def collectFace(self):
        cap=cv2.VideoCapture(0)
        detector=dlib.get_frontal_face_detector()
        predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        if not os.path.exists("faces"):
            os.makedirs("faces")

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

                crop_frame = frame[y1:y2, x1:x2]
                time=datetime.datetime.now()
                name="%s_%s_%s_%s_%s_%s.jpg"%(time.year, time.month, time.day, time.hour, time.minute, time.second)
                cv2.imwrite("./faces/"+name,crop_frame)
                cv2.putText(frame,t,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255),1)

            cv2.imshow("Please press 'q' to exit",frame)
            if cv2.waitKey(1)==ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        msg1=QtGui.QMessageBox()
        msg1.setWindowTitle("Faces collected")
        msg1.setInformativeText("Face files collection are complete! You can find your image files in the faces folder.")
        msg1.exec_()
# ----------------------------------------------------------------
    def facerecog(self):
        cap = cv2.VideoCapture(0)
        fr = face_recognition
        imgpath = "./facesData/"

        n=0
        names=[]
        faces=[]
        files = os.listdir(imgpath)

        for i in files:
            i=i.replace(".jpg","")
            names.append(i)

        lenf=len(files)

        for j in range(lenf):
            enc=fr.face_encodings(fr.load_image_file(imgpath+files[j]))[0]
            faces.append(enc)

        while True:
            ret, frame = cap.read()
            rgb_frame = frame[:, :, ::-1]

            face_locations = fr.face_locations(rgb_frame)
            face_encodings = fr.face_encodings(rgb_frame, face_locations)

            for (x1, y2, x2, y1), face_encoding in zip(face_locations, face_encodings):
                matches = fr.compare_faces(faces, face_encoding)

                showName = "Unknown"

                dist = fr.face_distance(faces, face_encoding)
                matchIndex = np.argmin(dist)
                if matches[matchIndex]:
                    showName = names[matchIndex]

                cv2.rectangle(frame, (y1,x1), (y2,x2), (0,255,0), 4)

                cv2.rectangle(frame, (y1,x2-30), (y2,x2), (255,0,0), cv2.FILLED)
                cv2.putText(frame, showName, (y1+10, x2-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

            cv2.imshow("Face Recognition Operating, Please press 'q' to exit", frame)
            if cv2.waitKey(1)==ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__=="__main__":
    app=QtGui.QApplication(sys.argv)
    w=Myapp()
    w.show()
    sys.exit(app.exec_())