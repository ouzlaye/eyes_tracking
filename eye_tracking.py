import numpy as np 
import cv2 

#read haarcascade and eye classifier from opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade= cv2.CascadeClassifier('haarcascade_eye.xml')
#------------------------------------------------------------
# initilaisation du deteteur de blob
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

# lecture de l'image avec opencv 
img = cv2.imread("test.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.3, 5)
# On cré une fonction pour la detection des yeux 
def detect_eyes( img,img_gray, lest, rest, classifier):
    leftEye= None
    rightEye = None
    leftEyeG = None
    rightEyeG = None
    coord = eye_cascade.detectMultiScale(img_gray, 1.3, 5)
    if coord is None or len(coord) == 0:
        pass

    else:
        for (x, y, w, h) in coord:
            eyecenter = int(float(x) + (float(w)/ float(2)))
            if lest[0] < eyecenter and eyecenter < lest[1]:
                leftEye = img[y:y + h, x:x +w]
                leftEyeG = img_gray[y:y +h, x:x + w]
                leftEye, leftEyeG  = cut_eyebrows(leftEye, leftEyeG)
            elif rest[0] < eyecenter and eyecenter < rest[1]:
                rightEye = img[y:y + h, x:x +w]
                rightEyeG = img_gray[y:y +h, x:x +w]
                rightEye, rightEyeG = cut_eyebrows(rightEye, rightEyeG)
            else:
                pass

    return leftEye, rightEye, leftEyeG, rightEyeG



    
# fonction de detection de face 
def face_detection(img, img_gray, classifier):
    
    coords =face_cascade.detectMultiScale(img, 1.3, 5)
    if len(coords )> 1:
        big = (0, 0, 0, 0)
        for i in coords:
            if i[3]  > big[3]:
                big = i
        big = np.array([i], np.int32)
    elif len(coords) == 1:
        big = coords
    else:
        return None, None, None ,None
    for (x, y, w, h) in big:
        frame = img[y:y +h, x:x+w]
        frame_gray = img_gray[y:y + h, x:x+w]
        lest = (int(w * 0.1), int(w * 0.45))
        rest = (int(w * 0.55), int(w * 0.9))
        X, Y = x, y 

    return frame, frame_gray, lest, rest, X, Y
#---------------------------------------------------
# fonction pour couper la partie de l'oeil depuis l'image
def cut_eyebrows(img, imgG):
    height, width = img.shape[:2]
    img = img[15:height, 0:width]
    imgG= imgG[15:height, 0:width]

    return img, imgG
#----------------------------------------------------------
def blob_process(img, threshold, detector, prevArea=None):
    
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations = 2)
    img = cv2.dilate(img, None, iterations= 4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    if keypoints and prevArea and len(keypoints) > 1 :
        tmp = 1000
        for keypoint in keypoints: 
            if abs(keypoint.size - prevArea)< tmp:
                ans = keypoint
                tmp = abs(keypoint.size - prevArea)
        keypoints = np.array(ans)
    return keypoints

#------------------------------------------------------
def nothing():

    pass


def draw_blobs():
    #afficher le blob
    cv2.drawKeypoints(img, keypoints, imgG, (0,0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#implementation de la fonction main

vc = cv2.VideoCapture(0)
cv2.namedWindow('image')
#cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

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
