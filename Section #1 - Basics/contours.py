#pylint:disable=no-member

from pathlib import Path

import cv2 as cv
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / 'Resources' / 'Photos' / 'astapogi.webp'

img = cv.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'Could not load image: {img_path}')

img = cv.resize(img, (700, 500))
cv.imshow('astapogi', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)