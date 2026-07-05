#pylint:disable=no-member

from pathlib import Path

import cv2 as cv
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / 'Resources' / 'Photos' / 'astapogi.webp'

img = cv.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'Could not load image: {img_path}')

scale_percent = 40
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

cv.imshow('Astapogi', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)

center = (img.shape[1] // 2, img.shape[0] // 2)
diamond_points = [
    (center[0], center[1] - 120),
    (center[0] + 120, center[1]),
    (center[0], center[1] + 120),
    (center[0] - 120, center[1]),
]

diamond_shape = np.zeros(img.shape[:2], dtype='uint8')
diamond_shape = cv.fillPoly(diamond_shape, [np.array(diamond_points, dtype=np.int32)], 255)
cv.imshow('Diamond Shape', diamond_shape)

masked = cv.bitwise_and(img, img, mask=diamond_shape)
cv.imshow('Weird Shaped Masked Image', masked)

cv.waitKey(0)
