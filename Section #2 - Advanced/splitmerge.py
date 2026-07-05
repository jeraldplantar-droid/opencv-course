#pylint:disable=no-member

from pathlib import Path

import cv2 as cv
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / 'Resources' / 'Photos' / 'astalarge.jpg'

img = cv.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'Could not load image: {img_path}')

cv.imshow('Astalarge', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b,g,r = cv.split(img)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])


cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)