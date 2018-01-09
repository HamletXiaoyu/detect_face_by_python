# coding:utf-8

import cv2

photo_path = './images/face.jpg'
classifier = './classifier/haarcascade_frontalface_default.xml'

#load image
image = cv2.imread(photo_path)

#binary
gary = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#get face training data
face_casacade = cv2.CascadeClassifier(classifier)

#detect face
faces = face_casacade.detectMultiScale(image)

color = (255, 0, 0)
stroke_weight = 1

print(len(faces))
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color, stroke_weight)
    cv2.imshow("face detection", image)

cv2.waitKey()

