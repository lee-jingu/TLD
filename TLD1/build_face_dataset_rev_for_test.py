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
	frame = imutils.resize(frame, width=1280)
	orig = frame.copy()

	cv2.imshow("Frame", frame)
	total += 1
	key = cv2.waitKey(1) & 0xFF

	if total%100==0:
		p = os.path.sep.join([args["output"], "{}.png".format(str(i).zfill(5))])
		cv2.imwrite(p, orig)
		i += 1

	elif key == ord("q"):
		break

print("{} face images stored".format(i))

cv2.destroyAllWindows()