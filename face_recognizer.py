#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
images = []
labels = []
# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createFisherFaceRecognizer()
def my_get(path, code):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    # images will contains face images
    # labels will contains the label that is assigned to the image
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        res = cv2.resize(image,(10,10))
        equ = cv2.equalizeHist(res)
        cv2.imshow("show", equ)
        cv2.waitKey(50)
        # Get the label of the image
        nbr = code
        images.append(equ)
        labels.append(nbr)
    return images, labels




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
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        print nbr
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            res = cv2.resize(image,(10,10))
            equ = cv2.equalizeHist(res)
            images.append(equ)
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...",equ)
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './yalefaces'

# Call the get_images_and_labels function and get the face images and the
# corresponding labels
get_images_and_labels(path)
#my_get('./datalinh',1000)
my_get('./crop/100-Linh',1000)
my_get('./crop/100-Quynh',2000)
my_get('./crop/100-Thanh',3000)
my_get('./crop/100-Quang',4000)
my_get('./crop/100-Tuan',5000)
my_get('./crop/100-Mai',6000)
my_get('./crop/100-Du',7000)
my_get('./crop/100-Quan',8000)
my_get('./crop/100-Viet',9000)
my_get('./crop/100-Mai',10000)
#dat train
my_get('./crop-train/yaleB01',100)
my_get('./crop-train/yaleB02',101)
my_get('./crop-train/yaleB03',102)
my_get('./crop-train/yaleB04',103)
my_get('./crop-train/yaleB05',104)
my_get('./crop-train/yaleB06',105)
my_get('./crop-train/yaleB07',106)
my_get('./crop-train/yaleB08',107)
my_get('./crop-train/yaleB09',108)
my_get('./crop-train/yaleB10',109)
my_get('./crop-train/yaleB11',110)
my_get('./crop-train/yaleB12',111)
my_get('./crop-train/yaleB13',112)
my_get('./crop-train/yaleB15',114)
my_get('./crop-train/yaleB16',115)
my_get('./crop-train/yaleB17',116)
my_get('./crop-train/yaleB18',117)
my_get('./crop-train/yaleB19',118)
my_get('./crop-train/yaleB20',119)
my_get('./crop-train/yaleB21',120)
my_get('./crop-train/yaleB22',121)
my_get('./crop-train/yaleB23',122)
my_get('./crop-train/yaleB34',123)
my_get('./crop-train/yaleB25',124)
my_get('./crop-train/yaleB26',125)
my_get('./crop-train/yaleB27',126)
my_get('./crop-train/yaleB28',127)
my_get('./crop-train/yaleB29',128)
my_get('./crop-train/yaleB30',129)
my_get('./crop-train/yaleB31',130)
my_get('./crop-train/yaleB32',131)
my_get('./crop-train/yaleB33',132)
my_get('./crop-train/yaleB34',133)
my_get('./crop-train/yaleB35',134)
my_get('./crop-train/yaleB36',135)
my_get('./crop-train/yaleB37',136)
my_get('./crop-train/yaleB38',137)
my_get('./crop-train/yaleB39',138)


my_get('./crop-train/train01',201)
my_get('./crop-train/train02',202)
my_get('./crop-train/train03',203)
my_get('./crop-train/train04',204)
my_get('./crop-train/train05',205)
my_get('./crop-train/train06',206)
my_get('./crop-train/train07',207)
my_get('./crop-train/train08',208)

cv2.destroyAllWindows()
# Perform the tranining
recognizer.train(images, np.array(labels))


recognizer.save('./data.xml')
print "Done!"
#
# Append the images with the extension .sad into image_paths
# image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
# for image_path in image_paths:
#     predict_image_pil = Image.open(image_path).convert('L')
#     predict_image = np.array(predict_image_pil, 'uint8')
#     faces = faceCascade.detectMultiScale(predict_image)
#     for (x, y, w, h) in faces:
#         nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
#         nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
#         if nbr_actual == nbr_predicted:
#             print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
#         else:
#             print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
#         cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
#         cv2.waitKey(1000)
