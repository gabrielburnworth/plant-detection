"""Weed Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import numpy as np
import cv2

def detect_weeds(image, **kwargs):
	"""Detect plants in image and saves an image with plants marked.
	   
	   Args:
		debug (boolean): output debug images
		morph (int): amount of filtering (default = 5)
	"""
	debug = False
	morph_amount = None
	for key in kwargs:
		if key == 'debug': debug = kwargs[key]
		if key == 'morph': morph_amount = kwargs[key]

	def imsavename(step, description):
		name = image[:-4]
		details = description
		if step is not None:
			details = '{}_{}'.format(step, details)
		if step > 2 or (debug and step is None):
			details = '{}_morph={}'.format(details, morph_amount)
		filename = '{}_{}.png'.format(name, details)
		print "Image saved: {}".format(filename)
		return filename
	
	print "\nProcessing image: {}".format(image)
	
	# Load image
	img = cv2.imread(image, 1)
	blur = cv2.medianBlur(img, 5)
	if debug:
	    cv2.imwrite(imsavename(0, 'blurred'), blur)

	# Decide color range to detect
	green = np.uint8([[[0, 255, 0]]])
	hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
	H = hsv_green[0, 0, 0]
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	bound = 30
	minimum = 50
	lower_green = np.array([H - bound, minimum, minimum])
	upper_green = np.array([H + bound, 255, 255])

	# Create weed mask
	mask = cv2.inRange(hsv, lower_green, upper_green)
	if debug:
	    cv2.imwrite(imsavename(1, 'mask'), mask)
	    res = cv2.bitwise_and(img, img, mask=mask)
	    cv2.imwrite(imsavename(2, 'masked'), res)

	# Process mask to try to make weeds more coherent
	if morph_amount is None: morph_amount = 5
	kernel = np.ones((morph_amount, morph_amount), np.uint8)
	proc = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	if debug:
		cv2.imwrite(imsavename(3, 'processed-mask'), proc)
		res2 = cv2.bitwise_and(img, img, mask=proc)
		cv2.imwrite(imsavename(4, 'processed-masked'), res2)

	# Find contours (hopefully of outside edges of weeds)
	contours, hierarchy = cv2.findContours(proc, 1, 2)
	print "{} plants detected in image.".format(len(contours))

	# Loop through contours
	for i in range(len(contours)):
		# Calculate weed location by using centroid of contour
		cnt = contours[i]
		M = cv2.moments(cnt)
		try:
			cx = int(M['m10'] / M['m00'])
			cy = int(M['m01'] / M['m00'])
		except ZeroDivisionError:
			continue
		if i == 0:
			print "Detected plant center pixel coordinates:"
		print "    x={} y={}".format(cx, cy)

		# Mark weed with red circle
		cv2.circle(img, (cx,cy), 20, (0,0,255), 4)

		if debug:
			cv2.drawContours(proc, [cnt], 0, (255,255,255), 3)

	if debug:
		cv2.imwrite('{}_5_contours.png'.format(image[:-4]), proc)

	# Save soil image with weeds marked
	cv2.imwrite(imsavename(None, 'marked'), img)

if __name__ == "__main__":
	image = "soil_image.jpg"
	detect_weeds(image, morph=15)
