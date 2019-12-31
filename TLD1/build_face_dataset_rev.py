from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-c", "--cascade", type=str, default="haarcascade_frontalface_alt.xml")
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-s", "--start", type=int, default="0")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])

print("starting video stream...")
vs = cv2.VideoCapture(args["input"])

time.sleep(2.0)
total = 0
i = args["start"]

while True:

	ret,frame = vs.read()
	frame = imutils.resize(frame, width=750)
	orig = frame.copy()
	gray = cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2GRAY)
	rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cropping = orig[y:y+h, x:x+w]
		cropping = imutils.resize(cropping, width = 500)

	cv2.imshow("Frame", frame)
	total += 1
	key = cv2.waitKey(1) & 0xFF

	if total%5==0 and len(rects)>0:
		p = os.path.sep.join([args["output"], "{}.png".format(str(i).zfill(1))])
		cv2.imwrite(p, cropping)
		i += 1

	elif key == ord("q"):
		break

print("{} face images stored".format(i))

cv2.destroyAllWindows()