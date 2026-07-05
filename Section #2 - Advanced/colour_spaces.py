#pylint:disable=no-member

from pathlib import Path

import cv2 as cv

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / 'Resources' / 'Photos' / 'astalarge.jpg'

img = cv.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'Could not load image: {img_path}')

cv.imshow('Astalarge', img)

# plt.imshow(img)
# plt.show()

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)

# HSV to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR', lab_bgr)

cv.waitKey(0)