import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageRegistry'
images = []
classNames = []
myList = os.listdir(path)
for c in myList:
    currentImg = cv2.imread(f'{path}/{c}')
    images.append(currentImg)
    classNames.append(os.path.splitext(c)[0])


def auto_encode(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def mark_registry(name):
    with open('register.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encode_list_known = auto_encode(images)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img_resize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    faces_current_frame = face_recognition.face_locations(img_resize)
    encodes_current_frame = face_recognition.face_encodings(img_resize, faces_current_frame)

    for encode_face, faceLoc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        faceDis = face_recognition.face_distance(encode_list_known, encode_face)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            mark_registry(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)
