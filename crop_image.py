import cv2, os
import numpy as np
from PIL import Image
import random


facePath = "./haarcascade_frontalface_default.xml"
faceProfilePath = "./haarcascade_profileface.xml"
smilePath = "./haarcascade_smile.xml"
eyesPath = "./haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(facePath)
faceProfileCascade = cv2.CascadeClassifier(faceProfilePath)
smileCascade = cv2.CascadeClassifier(smilePath)
eyesCascade = cv2.CascadeClassifier(eyesPath)

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    # labels will contains the label that is assigned to the image
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor= 1.2,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            sv = image[y:y+h, x:x+w]
            cv2.imwrite('./test/'+ str(random.randint(0,1000)) + 'x.png',sv)
    # return the images list and labels list

get_images_and_labels('./mai')
