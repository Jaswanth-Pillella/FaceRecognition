import cv2
# import numpy as np
import face_recognition

img1 = face_recognition.load_image_file('ImageRecog/Add-any-image.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = face_recognition.load_image_file('ImageRecog/Add-any-image.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

faceLoc1 = face_recognition.face_locations(img1)[0]
encode1 = face_recognition.face_encodings(img1)[0]
cv2.rectangle(img1, (faceLoc1[-3], faceLoc1[-4]), (faceLoc1[-1], faceLoc1[2]), (255, 0, 255), 2)

faceLoc2 = face_recognition.face_locations(img2)[0]
encode2 = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2, (faceLoc2[-3], faceLoc2[-4]), (faceLoc2[-1], faceLoc2[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encode1], encode2, 0.6)
faceDistance = face_recognition.face_distance([encode1], encode2)
cv2.putText(img2, f'{result} {round(faceDistance[0],2)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

cv2.imshow('img1-name', img1)
cv2.imshow('img2-name', img2)
cv2.waitKey(0)

#here ImageRecog/Add-any-image.jpg represents to add image to the specified path, create a folder named as ImageRecog and add your images.
# imshow function displays image in a new window
