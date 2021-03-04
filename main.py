import sys
import cv2
import numpy as np
import eye_tracking
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qtimer
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
#read haarcascade and eye classifier from opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml')
#------------------------------------------------------------
# initilaisation du deteteur de blob
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        loadUi('GUImain.ui', self)
        with open("style.css", "r") as css:
            self.setStyleSheet(css.read())

        self.face_detector= face_cascade
        self.eye_detector = eye_cascade
        self.detector = detector
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.camera_is_running = False 
        self.previous_right_keypoints = None
        self.previous_left_keypoints  =None
        self.previous_right_blob_area = None
        self.previous_left_blob_area = None


    def start_webcam(self):
        if not self.camera_is_running:
            self.capture = cv2.VideoCapture(cv2.CAP_DSHOW)
            if self.capture is None:
                self.capture = cv2.VideoCapture(0)
            self.camera_is_running = True
            self.timer = Qtimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(2)


    def stop_webcam(self):
        if self.camera_is_running:
            self.capture.release()
            self.timer.stop()
            self.camera_is_running = not self.camera_is_running

    
    def update_frame(self):
        _,base_image = self.capture.read()
        self.displaye_image(base_image)

        processed_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

        face_frame, face_frame_gray, leep, reep, _, _ = eye_tracking.face_detection(base_image, processed_image, self.face_detector )
            
        if face_frame is not None:

            left_eye_frame, right_eye_frame, left_eye_frame_gray, right_eye_frame_gray = eye_tracking.detect_eyes(
                face_frame, face_frame_gray, leep, reep, self.eye_detector)
                
            if right_eye_frame is not None:
                if self.rightEyeCheckbox.isChecked():
                    right_eye_threshold = self.rightEyeTrheshold.value()
                    right_keypoints, self.previous_right_keypoints, self.previous_right_blob_area = self.get_keypoints(
                    right_eye_frame, right_eye_frame_gray, right_eye_threshold, previous_area = self.previous_right_blob_area, previous_keypoint= self.previous_right_keypoints)

                    eye_tracking.draw_blobs(right_eye_frame, right_keypoints)
                    right_eye_frame = np.require(right_eye_frame, np.uint8, 'C')
                    self.displaye_image(right_eye_frame, window = 'right')
            if self.pupilsCheckbox.isChecked():
                self.displaye_image(base_image)

    def get_keypoints(self, frame, frame_gray, threshold, previous_keypoint, previous_area):
        
        keypoints =eye_tracking.blob_process(frame_gray, threshold, self.detector, prevArea= previous_area)

        if keypoints:
            previous_keypoint = keypoints
            previous_area = keypoints[0].size

        else:
            keypoints= previous_keypoint
        
        return keypoints, previous_keypoint, previous_area

    
    def displaye_image(self, img, window='main'):

        qformat = QImage_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat = QImage.Format_RGBA8888

            else:
                qformat = QImage.Format_RGB888 
        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()
        if window == 'main':
            self.baseImage.setPixmap(QPixmap.fromImage(out_image))
            self.baseImage.setScaledContents(True)

        if window == 'left':
            self.leftEyeBox.setPixmap(QPixmap.fromImage(out_image))
            self.leftEyeBox.setScaledContents(True)

        if window =='right':
            self.rightEyeBox.setPixmap(QPixmap.fromImage(out_image))
            self.rightEyeBox.setScaledContents(True)


if __name__=="__main__":
    app=QApplication(sys.argv)
    window= Window()
    window.setWindowTitle("Mon Image")
    window.show()
    sys.exit(app.exec_())



