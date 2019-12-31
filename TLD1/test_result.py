import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True)
ap.add_argument("-f", "--encodings2", default="enc2.pickle")
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", type=str)
ap.add_argument("-y", "--display", type=int, default=1)
ap.add_argument("-d", "--detection-method", type=str, default="cnn")
args = vars(ap.parse_args())

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
data_test= pickle.loads(open(args["encodings2"],"rb").read())
print("[INFO] processing video...")
stream = cv2.VideoCapture(args["input"])
writer = None

while True:
	(grabbed, frame) = stream.read()
	frame = imutils.resize(frame, width=1000)
	if not grabbed:
		break

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=1000)
	r = frame.shape[1] / float(rgb.shape[1])

	boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	names2 = []
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],	encoding)
		matches2 = face_recognition.compare_faces(data_test["encodings"],encoding)
		name = "Unknown"
		name2 = "Unknown"
		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			counts2 = {}
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
				for j in matchedIdxs:
					name2 = data["names"][j]
					counts2[name2] = counts2.get(name2, 0) + 1
				if counts != counts2 and i==0:
					print("counts1 : {}\n, counts2 = {}".format(counts,counts2))
			name = max(counts, key=counts.get)
			names.append(name)
			name2 = max(counts2, key=counts.get)
			names2.append(name2)

	for ((top, right, bottom, left), name) in zip(boxes, names):
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 24,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces t odisk
	if writer is not None:
		writer.write(frame)

	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# close the video file pointers
stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()