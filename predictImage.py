import cv2, os
import numpy as np
from PIL import Image

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createFisherFaceRecognizer()


recognizer.load('./data.xml')

facePath = "./haarcascade_frontalface_default.xml"
faceProfilePath = "./haarcascade_profileface.xml"
smilePath = "./haarcascade_smile.xml"
eyesPath = "./haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(facePath)
faceProfileCascade = cv2.CascadeClassifier(faceProfilePath)
smileCascade = cv2.CascadeClassifier(smilePath)
eyesCascade = cv2.CascadeClassifier(eyesPath)

while True:
    print 'start!'
    image_pil = Image.open('./image/IMG_20170617_131154.jpg')
        # Convert the image format into numpy array
    image = np.array(image_pil, 'uint8')
    frame =  image# Capture frame-by-frame
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    sF = 1.2
    # mirror image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = equ[y:y+h, x:x+w]
        prediction = recognizer.predict(cv2.resize(roi_gray,(10,10)))
        cv2.putText(frame, str(prediction[0]),
           (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        print prediction
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
        roi_gray = equ[y:y+h, x:x+w]
        prediction = recognizer.predict(cv2.resize(roi_gray,(10,10)))
        cv2.putText(frame, str(prediction[0]),
           (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        print prediction
    #
    frame_ = cv2.flip(frame,1)
    gray_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    equ_ = cv2.equalizeHist(gray_)
    facesProfile_ = faceProfileCascade.detectMultiScale(
        gray_,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in facesProfile_:
        cv2.rectangle(frame_, (x, y), (x+w, y+h), (0, 0, 50), 2)
        roi_gray = equ_[y:y+h, x:x+w]
        prediction = recognizer.predict(cv2.resize(roi_gray,(10,10)))
        cv2.putText(frame_, str(prediction[0]),
           (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        print prediction






    #cv2.cv.Flip(frame, None, 1)

    cv2.imshow('Smile Detector', cv2.resize(gray,(500,500)))
    c = cv2.cv.WaitKey(7) % 0x100
