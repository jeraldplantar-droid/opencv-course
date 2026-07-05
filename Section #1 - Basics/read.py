#pylint:disable=no-member

from pathlib import Path

import cv2 as cv

base_dir = Path(__file__).resolve().parent.parent
img_path = base_dir / 'Resources' / 'Photos' / 'astalarge.jpg'
video_path = base_dir / 'Resources' / 'Videos' / 'astaVid.mp4'

img = cv.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'Could not load image: {img_path}')

cv.imshow('astalarge', img)
cv.waitKey(0)

# Reading Videos
capture = cv.VideoCapture(str(video_path))
if not capture.isOpened():
    raise FileNotFoundError(f'Could not open video: {video_path}')

while True:
    isTrue, frame = capture.read()

    # if cv.waitKey(20) & 0xFF==ord('d'):
    # This is the preferred way - if `isTrue` is false (the frame could
    # not be read, or we're at the end of the video), we immediately
    # break from the loop.
    if isTrue:
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()
