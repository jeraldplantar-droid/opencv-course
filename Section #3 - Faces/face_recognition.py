#pylint:disable=no-member

import argparse
from pathlib import Path

import cv2 as cv

base_dir = Path(__file__).resolve().parent.parent
haar_path = Path(__file__).resolve().parent / 'haar_face.xml'
default_image_path = base_dir / 'Resources' / 'Faces' / 'val' / 'elton_john' / '1.jpg'

parser = argparse.ArgumentParser(description='Show detected faces in an image.')
parser.add_argument('image', nargs='?', default=str(default_image_path), help='Path to the image you want to use')
args = parser.parse_args()

image_path = Path(args.image).expanduser()
if not image_path.is_absolute():
    image_path = (Path(__file__).resolve().parent / image_path).resolve()

haar_cascade = cv.CascadeClassifier(str(haar_path))
if haar_cascade.empty():
    raise FileNotFoundError(f'Could not load cascade: {haar_path}')

img = cv.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f'Could not load image: {image_path}')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

if len(faces_rect) == 0:
    print('No face detected in the image.')
else:
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
