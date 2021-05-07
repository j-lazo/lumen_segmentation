# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2
import time


"""frame_rate = 60
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 300))

while (cap.isOpened()):
    prev = 0
    time_elapsed = time.time() - prev
    res, image = cap.read()
    print(time_elapsed, 1/frame_rate)
    if res == True:
        if time_elapsed > 1./frame_rate:
            resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', image)
            print('True')
            prev = time.time()"""

cap = cv2.VideoCapture(0)
frame_rate = 60
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    prev = 0
    # Capture frame-by-frame
    ret, frame = cap.read()
    time_elapsed = time.time() - prev
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if time_elapsed > 1. / frame_rate:
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()