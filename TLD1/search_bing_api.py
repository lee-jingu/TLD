from requests import exceptions
import argparse
import requests
import cv2
import os
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str, default="haarcascade_frontalface_alt.xml")
ap.add_argument("-q", "--query", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])

API_KEY = "9e866d479822497e97c8f84d7c66e15f"
MAX_RESULTS = 1000
GROUP_SIZE = 1000
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
	term))

total = 0

for offset in range(0, estNumResults, GROUP_SIZE):
	print("[INFO] making request for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
	params["offset"] = offset
	search = requests.get(URL, headers=headers, params=params)
	search.raise_for_status()
	results = search.json()
	print("[INFO] saving images for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))

	# loop over the results
	for v in results["value"]:
		# try to download the image
		try:
			# make a request to download the image
			print("[INFO] fetching: {}".format(v["contentUrl"]))
			r = requests.get(v["contentUrl"], timeout=30)

			# build the path to the output image
			ext = v["contentUrl"][v["contentUrl"].rfind("."):]
			p = os.path.sep.join([args["output"], "{}{}".format(
				str(total).zfill(5), ext)])

			# write the image to disk
			f = open(p, "wb")
			f.write(r.content)
			f.close()
			img = cv2.imread(p)
			orig = img.copy()
			gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
			rects = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
			for (x, y, w, h) in rects:
				cropping = orig[y:y+h, x:x+w]
				cropping = imutils.resize(cropping, width = 500)
			if len(rects)==1:
				cv2.imwrite(p, cropping)

		# catch any errors that would not unable us to download the
		# image
		except Exception as e:
			# check to see if our exception is in our list of
			# exceptions to check for
			if type(e) in EXCEPTIONS:
				print("[INFO] skipping: {}".format(v["contentUrl"]))
				continue

		# try to load the image from disk
		image = cv2.imread(p)

		# if the image is `None` then we could not properly load the
		# image from disk (so it should be ignored)
		if image is None or len(rects) != 1:
			print("[INFO] deleting: {}".format(p))
			os.remove(p)
			continue

		# update the counter
		total += 1