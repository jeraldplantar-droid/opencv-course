from pathlib import Path
import cv2 as cv
base_dir = Path(r'C:/jerald/opencv-course')
image_path = base_dir / 'Resources' / 'Faces' / 'val' / 'mikerey_pogi' / 'MIkerey.jpg'
haar_path = base_dir / 'Section #3 - Faces' / 'haar_face.xml'
print('image exists', image_path.exists())
img = cv.imread(str(image_path))
print('img None', img is None, 'shape', None if img is None else img.shape)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar = cv.CascadeClassifier(str(haar_path))
print('cascade empty', haar.empty())
for params in [
    {'scaleFactor':1.1,'minNeighbors':4,'minSize':(30,30)},
    {'scaleFactor':1.05,'minNeighbors':4,'minSize':(20,20)},
    {'scaleFactor':1.05,'minNeighbors':3,'minSize':(20,20)},
    {'scaleFactor':1.03,'minNeighbors':3,'minSize':(20,20)},
    {'scaleFactor':1.01,'minNeighbors':3,'minSize':(20,20)}
]:
    faces = haar.detectMultiScale(gray, **params)
    print(params, 'count', len(faces), faces)
