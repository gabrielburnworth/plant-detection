#!/usr/bin/env python
import sys
import cv2
from Plant_Detection import Plant_Detection

if len(sys.argv) == 1:
    filename = 'soil_image.jpg'
else:
    filename = sys.argv[1]
window = 'Plant Detection'
HSVwindow = 'HSV Selection'
override_HSV_defaults = 0
HSVwindow_loaded = 0
HSV_bounds = [[30, 20, 20], [90, 255, 255]]

def HSV_trackbar_name(P, bound):
    if P == 'H': P = 'Hue'
    if P == 'S': P = 'Saturation'
    if P == 'V': P = 'Value'
    return '{} {} {}'.format(P, bound, ' ' * (12 - len(P)))

def process(x):
    global HSV_bounds
    global override_HSV_defaults
    global HSVwindow_loaded

    HSVwindow_open = cv2.getTrackbarPos('Open HSV Selection Window', window)
    if HSVwindow_open and not HSVwindow_loaded:
        pass
    else:
        # Get parameter values
        blur = cv2.getTrackbarPos('Blur', window)
        if blur % 2 == 0: blur += 1
        morph = cv2.getTrackbarPos('Morph', window)
        iterations = cv2.getTrackbarPos('Iterations', window)

        if HSVwindow_open:
            # get HSV values
            for b, bound in enumerate(['min', 'max']):
                for P in range(0, 3):
                    HSV_bounds[b][P] = cv2.getTrackbarPos(
                        HSV_trackbar_name('HSV'[P], bound),
                        HSVwindow)

        # Process image with parameters
        if override_HSV_defaults:
            PD = Plant_Detection(image=filename,
                  blur=blur, morph=morph, iterations=iterations,
                  HSV_min=HSV_bounds[0], HSV_max=HSV_bounds[1],
                  debug=True, save=False)
            PD.detect_plants()
            img = PD.final_debug_image
        else:
            PD = Plant_Detection(image=filename,
                  blur=blur, morph=morph, iterations=iterations,
                  debug=True, save=False)
            PD.detect_plants()
            img = PD.final_debug_image

        #Show processed image
        cv2.imshow(window, img)

def HSV_selection(open_window):
    global HSV_bounds
    global override_HSV_defaults
    global HSVwindow_loaded
    override_HSV_defaults = 1

    if open_window:
        cv2.namedWindow(HSVwindow)
        for b, bound in enumerate(['min', 'max']):
            for P, limit in zip(range(0, 3), [179, 255, 255]):
                cv2.createTrackbar(
                    HSV_trackbar_name('HSV'[P], bound),
                    HSVwindow, 0, limit, process)
                cv2.setTrackbarPos(
                    HSV_trackbar_name('HSV'[P], bound),
                    HSVwindow, HSV_bounds[b][P])
        HSVwindow_loaded = 1
    else: # close window
        cv2.destroyWindow(HSVwindow)
        HSVwindow_loaded = 0

cv2.namedWindow(window)
cv2.createTrackbar('Blur', window, 0, 100, process)
cv2.createTrackbar('Morph', window, 1, 100, process)
cv2.createTrackbar('Iterations', window, 1, 100, process)
cv2.createTrackbar('Open HSV Selection Window', window, 0, 1, HSV_selection)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
