#!/usr/bin/python

import dlib, cv2
import numpy as np
import requests

predictor_path = 'models/shape_predictor_68_face_landmarks.dat'

url = 'https://github.com/22preich/-Deprehendere/blob/showcase2020/models/shape_predictor_68_face_landmarks.dat' \
      '?raw=true '

r = requests.get(url, allow_redirects=True)

open(predictor_path, 'wb').write(r.content)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)


def crop(face_file_path):
    img = cv2.imread(face_file_path)

    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        return False, None

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    images = dlib.get_face_chips(img, faces, size=160)

    ret = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    return img, cv2.imencode('.jpg', img)[1]


def cropFileStorageObject(file_storage):
    filestr = file_storage.read()

    npimg = np.fromstring(filestr, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(file_storage))
        return False, None

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    # images = dlib.get_face_chips(img, faces, size=160)

    # TODO we can choose faces
    # ret = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    return img, cv2.imencode('.jpg', img)[1]
