from pathlib import Path
import cv2 as cv
import numpy as np
base_dir = Path(r'C:/jerald/opencv-course')
image_path = base_dir / 'Resources' / 'Faces' / 'val' / 'mikerey_pogi' / 'MIkerey (2).jpg'
haar_path = base_dir / 'Section #3 - Faces' / 'haar_face.xml'
img = cv.imread(str(image_path))
if img is None:
    raise SystemExit('failed load image')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar = cv.CascadeClassifier(str(haar_path))
f = haar.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20,20), maxSize=(800,800))
output = img.copy()
for i,(x,y,w,h) in enumerate(f):
    color = (255,0,0)
    cv.rectangle(output, (x,y), (x+w,y+h), color, 2)
    cv.putText(output, str(i), (x, max(y-10,0)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
if len(f)>0:
    x,y,w,h = max(f, key=lambda rect: (rect[2]*rect[3], -abs(rect[2]/rect[3]-1)))
    cv.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 4)
    cv.putText(output, 'chosen', (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv.imwrite(str(base_dir / 'Section #3 - Faces' / 'mikerey2_debug.png'), output)
print('wrote debug image, count', len(f), 'chosen', (x,y,w,h) if len(f)>0 else None)
