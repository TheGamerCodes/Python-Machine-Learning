import cv2
image = cv2.imread("palm.jpg")

#View Palm image:
cv2.imshow("palm",image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,
cv2.COLOR_BGR2GRAY)

#Using Canny Edge Detector:
edges = cv2.Canny(gray,40,55,apertureSize = 3)
cv2.imshow("edges in palm",edges)
cv2.waitKey(0)

#Revert Colors:
edges=cv2.bitwise_not(edges)

#Blend Both Images:
cv2.imwrite("palmlines.jpg",edges)
palmlines = cv2.imread("palmlines.jpg")
img = cv2.addWeighted(palmlines, 0.3, image, 0.7,0)

'''
Palm Line Detection with OpenCV:
#TheGamerCodes
'''