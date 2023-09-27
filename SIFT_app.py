#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np


class My_App(QtWidgets.QMainWindow):

    #def __init__(self):
        #super(My_App, self).__init__()
        #loadUi("./SIFT_app.ui", self)
        #self.browse_button.clicked.connect(self.SLOT_browse_button)
   
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)
        # Timer used to trigger the camera

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

 
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]
        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
       	print("Loaded template image fie:" + self.template_path)

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):

        ret, frame = self._camera_device.read()
            #TODO run SIFT on the captured frame

        #SIFT camera frame 
        gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        sift_frame = cv2.SIFT_create()
        kp_frame, desc_frame = sift_frame.detectAndCompute(gray_frame,None)
        img_frame=cv2.drawKeypoints(gray_frame,kp_frame,frame)

        if hasattr(self, "template_path"):

            #SIFT chosen image
            img = cv2.imread(self.template_path)
            #img = cv2.imread("/home/fizzer/Downloads/bender.png") specified file
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp_img, desc_img = sift.detectAndCompute(gray,None)
            img=cv2.drawKeypoints(gray,kp_img,img)

            #Feature match 
            index_params = dict(algorithm = 1, trees =5) # type of alorithm, # recursive search num  
            search_params = dict()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc_img, desc_frame, k=2) #k best matches 
        
            #from best 2 matches compare distance to evaluate match quality
            gp=[]
            for m,n in matches: 
                if m.distance < .7*n.distance:
                    gp.append(m)
            
            #display best connections between frame and img
            connect = cv2.drawMatches(img,kp_img,img_frame,kp_frame, gp, img_frame)

            #Homography
            if len(gp)> 10:
                qry_pts = np.float32([kp_img[m.queryIdx].pt for m in gp]).reshape(-1,1,2)
                trn_pts = np.float32([kp_frame[m.trainIdx].pt for m in gp]).reshape(-1,1,2)
                
                matrix, mask = cv2.findHomography(qry_pts,trn_pts, cv2.RANSAC, 5.0)
                mask_matches = mask.ravel().tolist()

                #Transfrom perspective 
                h, w = gray.shape
                pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
                persp = cv2.perspectiveTransform(pts,matrix)
                
                colorbgr = (0,255,0)
                thickness = 3
                homography = cv2.polylines(frame, [np.int32(persp)], True, colorbgr, thickness)
                pixmap = self.convert_cv_to_pixmap(homography)
            else:
                pixmap = self.convert_cv_to_pixmap(connect)

            #Display frame 
        else:
            pixmap = self.convert_cv_to_pixmap(img_frame)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
                self._timer.stop()
                self._is_cam_enabled = False
                self.toggle_cam_button.setText("&Enable camera")
        else:
                self._timer.start()
                self._is_cam_enabled = True
                self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
