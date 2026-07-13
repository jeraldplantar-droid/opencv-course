#pylint:disable=no-member

import argparse
from pathlib import Path

import cv2 as cv

base_dir = Path(__file__).resolve().parent.parent
haar_path = Path(__file__).resolve().parent / 'haar_face.xml'
default_image_path = base_dir / 'Resources' / 'Faces' / 'val' / 'mikerey_pogi' / 'MIkerey (2).jpg'

parser = argparse.ArgumentParser(description='Show detected faces in an image.')
parser.add_argument('image', nargs='?', default=str(default_image_path), help='Path to the image you want to use')
args = parser.parse_args()


def resize_to_fit(image, max_width=900, max_height=700):
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image
    scale = min(max_width / w, max_height / h)
    return cv.resize(image, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

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
resized_gray = resize_to_fit(gray)
resized_img = resize_to_fit(img)

cv.imshow('Person', resized_gray)

faces_rect = haar_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(80, 80),
    maxSize=(1200, 1200)
)

if len(faces_rect) == 0:
    print('No face detected in the image.')
else:
    def face_score(rect):
        x, y, w, h = rect
        cx = x + w / 2
        cy = y + h / 2
        area = w * h
        aspect = w / h if h else 0
        center_penalty = abs(cx - img.shape[1] / 2)
        upper_bonus = -cy
        return (area, -abs(aspect - 1), upper_bonus, -center_penalty)

    x, y, w, h = max(faces_rect, key=face_score)

    def expand_rect(rect, scale=1.8, top_shift=0.25):
        x, y, w, h = rect
        cx = x + w / 2
        cy = y + h / 2
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_x = int(max(0, cx - new_w / 2))
        new_y = int(max(0, cy - new_h / 2) - h * top_shift)
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, img.shape[1] - new_x)
        new_h = min(new_h, img.shape[0] - new_y)
        return new_x, new_y, new_w, new_h

    x, y, w, h = expand_rect((x, y, w, h), scale=1.8, top_shift=0.25)
    cv.rectangle(
        resized_img,
        (int(x * resized_img.shape[1] / img.shape[1]), int(y * resized_img.shape[0] / img.shape[0])),
        (int((x + w) * resized_img.shape[1] / img.shape[1]), int((y + h) * resized_img.shape[0] / img.shape[0])),
        (0, 255, 0),
        thickness=2,
    )
    print(f'Number of face candidates found = {len(faces_rect)}')
    print(f'Chosen face rectangle = {x},{y},{w},{h}')

cv.imshow('Detected Face', resized_img)

cv.waitKey(0)
