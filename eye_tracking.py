import numpy as np 
import cv2 

#read haarcascade and eye classifier from opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml')
#------------------------------------------------------------
# initilaisation du deteteur en utilisant blob
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

# lecture de l'image avec opencv 
img = cv2.imread("test.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.3, 5)
# On cré une fonction pour la detection des yeux 
def detect_eyes( img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coord = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
    height = np.size(img, 0)
    width = np.size(img, 1)
    left_eye = None
    right_eye = None
    for (x, y, w, h)in coord:
        if y+h > height/2:
            pass

        eye_center = x+ w /2
        if eye_center  < width * 0.5:
            left_eye = img[y:y+h, x:x+w]
        else:
            right_eye = img[y:y+h, x:x+w]

    return left_eye, right_eye

# fonction de detection de face 
def face_detection(img, classifier):
    gray_frame= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords =face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords )> 1:
        big = (0, 0, 0, 0)
        for i in coords:
            if i[3]  > big[3]:
                big = i
        big = np.array([i], np.int32)
    elif len(coords) == 1:
        big = coords
    else:
        return None
    for (x, y, w, h) in big:
        frame = img[y:y +h, x:x+w]
    return frame
#---------------------------------------------------
# fonction pour couper la partie de l'oeil depuis l'image
def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]

    return img
#----------------------------------------------------------
def blob_process(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 42, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations = 2)
    img = cv2.dilate(img, None, iterations= 4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints

#------------------------------------------------------
def nothing():
    pass


#implementation de la fonction main

vc = cv2.VideoCapture(0)
cv2.namedWindow('image')
#cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
while True:
    _, frame = vc.read()
    face_frame = face_detection(frame, face_cascade)
    if face_frame is not None:
        eyes = detect_eyes(face_frame, eye_cascade)
        for eye in eyes:
            if eye is not None:
                #threshold = cv2.getTrackbarPos('threshold','image')
                eye = cut_eyebrows(eye)
                keypoints= blob_process(eye, detector)
                eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    else:
        print(" No face")    
    cv2.imshow('Image', frame)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"):
        break

vc.release()
cv2.destroyAllWindows()
"""
#draw face in images 
for (x, y, w, h) in face:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 0)
    gray_face = gray[y:y+h, x:x+w]
    faces= img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray_face)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(faces,(ex,ey), (ex+ew, ey+eh),(0,255,255), 2)
        cv2.imshow('Image', img)
        key=  cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
cv2.destroyAllWindows():
"""
