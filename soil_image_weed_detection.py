"""Weed Detection.

Detects green weeds on a dirt background
 and marks them with red circles.
"""
import numpy as np
import cv2

debug = False

# Load image
img = cv2.imread('soil_image.jpg',1)
blur = cv2.medianBlur(img, 5)
if debug:
	cv2.imwrite('0_blurred.png', blur)

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
	cv2.imwrite('1_mask.png', mask)
	res = cv2.bitwise_and(img, img, mask= mask)
	cv2.imwrite('2_masked.png', res)

# Process mask to try to make weeds more coherent
morph_amount = 50
kernel = np.ones((morph_amount, morph_amount), np.uint8)
proc = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
if debug:
	cv2.imwrite('3_processed_mask.png', proc)
	res2 = cv2.bitwise_and(img, img, mask= proc)
	cv2.imwrite('4_processed-masked.png', res2)

# Find contours (hopefully of outside edges of weeds)
contours,hierarchy = cv2.findContours(proc, 1, 2)
print "{} weeds detected.".format(len(contours))

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
	print "Weed detected at image pixel x={} y={}".format(cx, cy)

	# Mark weed with red circle
	cv2.circle(img,(cx,cy), 20, (0,0,255), 4)

	if debug:
		cv2.drawContours(proc, [cnt], 0, (255,255,255), 3)

if debug:
	cv2.imwrite('5_contours.png',proc)

# Save soil image with weeds marked
cv2.imwrite('marked.png',img)
