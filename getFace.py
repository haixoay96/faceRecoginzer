import cv2, os
import numpy as np
from PIL import Image
import random


# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()


recognizer.load('./data.xml')

facePath = "./haarcascade_frontalface_default.xml"
faceProfilePath = "./haarcascade_profileface.xml"
smilePath = "./haarcascade_smile.xml"
eyesPath = "./haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(facePath)
faceProfileCascade = cv2.CascadeClassifier(faceProfilePath)
smileCascade = cv2.CascadeClassifier(smilePath)
eyesCascade = cv2.CascadeClassifier(eyesPath)

cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

sF = 1.2

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    frame = cv2.flip(frame,1) # mirror image
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    # ---- Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        equ = cv2.equalizeHist(roi_gray)
        cv2.imwrite('./test/'+ str(random.randint(0,1000)) + '.png',equ)
        prediction = recognizer.predict(roi_gray)
        cv2.putText(frame, 'linh',
           (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        print prediction






    #cv2.cv.Flip(frame, None, 1)

    cv2.imshow('Smile Detector', frame)
    c = cv2.cv.WaitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
