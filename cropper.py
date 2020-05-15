#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool for image alignment.
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys

import dlib, cv2
import numpy as np

'''if len(sys.argv) != 3:
    print(
        "Call this program like this:\n"
        "   ./face_alignment.py shape_predictor_5_face_landmarks.dat ../examples/faces/bald_guys.jpg\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n")
    exit()'''

predictor_path = 'resources/shape_predictor_68_face_landmarks.dat'
# face_file_path = 'dbs/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)


def crop(face_file_path):
    # Load the image using Dlib
    # img = dlib.load_rgb_image(face_file_path)
    img = cv2.imread(face_file_path)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(face_file_path))
        return False, None

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    # window = dlib.image_window()

    # Get the aligned face images
    # Optionally:
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(img, faces, size=160)
    # window.set_image(image)

    # TODO we can choose faces
    # cv2.imshow("img", images[0])
    # cv2.waitKey()
    # cv2.imwrite(output_path, images[0])
    ret = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    return img, cv2.imencode('.jpg', img)[1]


def cropFileStorageObject(fileStorage):
    # Load the image using Dlib
    # img = dlib.load_rgb_image(face_file_path)
    # read image file string data
    filestr = fileStorage.read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(fileStorage))
        return False, None

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    # window = dlib.image_window()

    # Get the aligned face images
    # Optionally:
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(img, faces, size=160)
    # window.set_image(image)

    # TODO we can choose faces
    # cv2.imshow("img", images[0])
    # cv2.waitKey()
    # cv2.imwrite(output_path, images[0])
    ret = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    return img, cv2.imencode('.jpg', img)[1]


if __name__ == '__main__':
    pass
    # crop(face_file_path)

# It is also possible to get a single chip
# image = dlib.get_face_chip(img, faces[0])
# window.set_image(image)
# dlib.hit_enter_to_continue()
