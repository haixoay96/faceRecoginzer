import cv2, os
import numpy as np
from PIL import Image
import random
facePath = "./haarcascade_frontalface_default.xml"
faceProfilePath = "./haarcascade_profileface.xml"
faceCascade = cv2.CascadeClassifier(facePath)
faceProfileCascade = cv2.CascadeClassifier(faceProfilePath)

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

sF = 1.2
while True:

    ret, frame = cap.read() # Capture frame-by-frame
# mirror image
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',roi_gray)

    # ---- Draw a rectangle around the faces
    facesProfile = faceProfileCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55,55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in facesProfile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 100), 2)
        roi_gray = gray[y:y+h, x:x+w]
        cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',roi_gray)
    #
    frame_ = cv2.flip(frame,1)
    gray_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    facesProfile_ = faceProfileCascade.detectMultiScale(
        gray_,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in facesProfile_:
        cv2.rectangle(frame_, (x, y), (x+w, y+h), (0, 0, 50), 2)
        roi_gray = gray_[y:y+h, x:x+w]
        cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',roi_gray)

    cv2.imshow('Smile Detector', frame_)
    c = cv2.cv.WaitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
