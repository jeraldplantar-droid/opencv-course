#pylint:disable=no-member

from pathlib import Path

import cv2 as cv

base_dir = Path(__file__).resolve().parent.parent
video_path = base_dir / 'Resources' / 'Videos' / 'astaVid.mp4'


def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def changeRes(width, height):
    # Live video
    capture.set(3, width)
    capture.set(4, height)


# Reading Videos
capture = cv.VideoCapture(str(video_path))
if not capture.isOpened():
    raise FileNotFoundError(f'Could not open video: {video_path}')

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    frame_resized = rescaleFrame(frame, scale=.2)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()