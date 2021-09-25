#import the libraries:
import cv2
from matplotlib import pyplot as plt

#read image containing a face:
image1 = cv2.imread("nate_pics/nate1.png")

#Convert image1 to GrayScale:
gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

cv2.imshow("1st gray image", gray_image)
cv2.waitKey(0)

#Applying Haar Cascade over image1:
face_haar = cv2.CascadeClassifier("haarcascade_files/haarcascade_frontalface_alt.xml")
faces = face_haar.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

#To Draw a Triangle over the face:
for (x,y,w,h) in faces:
    cv2.rectangle(image1, (x,y), (x+w, y+h), (0,255,0), 2)

#Display image1 with face detected:
cv2.imshow("detected_image", image1) 
cv2.waitKey(0)


'''
Face Detection with OpenCV:
#TheGamerCodes
'''