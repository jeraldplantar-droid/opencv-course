from pathlib import Path
import cv2 as cv
base_dir = Path(r'C:/jerald/opencv-course')
image_path = base_dir / 'Resources' / 'Faces' / 'val' / 'mikerey_pogi' / 'MIkerey.jpg'
haar_path = base_dir / 'Section #3 - Faces' / 'haar_face.xml'
img = cv.imread(str(image_path))
if img is None:
    raise SystemExit('failed load')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar = cv.CascadeClassifier(str(haar_path))
def score(rect):
    x,y,w,h = rect
    aspect = w/h if h else 0
    area = w*h
    # prefer square faces and moderate size
    aspect_penalty = abs(aspect - 1)
    return area - aspect_penalty * 1000

faces = haar.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30,30), maxSize=(700,700))
print('raw count', len(faces), faces)
faces = sorted(faces, key=score, reverse=True)
for rect in faces[:10]:
    x,y,w,h = rect
    print('rect', rect, 'score', score(rect), 'area', w*h, 'aspect', w/h)
if faces:
    rect = faces[0]
    x,y,w,h = rect
    print('chosen', rect)
