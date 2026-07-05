#pylint:disable=no-member

import os
from pathlib import Path

import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
base_dir = Path(__file__).resolve().parent.parent
DIR = base_dir / 'Resources' / 'Faces' / 'train'
haar_path = Path(__file__).resolve().parent / 'haar_face.xml'

haar_cascade = cv.CascadeClassifier(str(haar_path))
if haar_cascade.empty():
    raise FileNotFoundError(f'Could not load cascade: {haar_path}')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(str(DIR), person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

output_dir = Path(__file__).resolve().parent
face_recognizer.save(str(output_dir / 'face_trained.yml'))
np.save(output_dir / 'features.npy', features)
np.save(output_dir / 'labels.npy', labels)
