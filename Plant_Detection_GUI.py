#!/usr/bin/env python
import sys
import os
import json
import cv2
from Plant_Detection import Plant_Detection

directory = os.path.dirname(os.path.realpath(__file__)) + os.sep

if len(sys.argv) == 1:
    filename = directory + 'soil_image.jpg'
else:
    filename = sys.argv[1]
window = 'Plant Detection'
HSVwindow = 'HSV Selection'
override_HSV_defaults = 0
HSVwindow_loaded = 0
try:  # Load input parameters from file
    with open(directory + "plant-detection_inputs.json", 'r') as f:
        inputs = json.load(f)
    blur_amount = inputs['blur']
    morph_amount = inputs['morph']
    iterations = inputs['iterations']
    HSV_min = [inputs['H'][0], inputs['S'][0], inputs['V'][0]]
    HSV_max = [inputs['H'][1], inputs['S'][1], inputs['V'][1]]
    HSV_bounds = [HSV_min, HSV_max]
    from_file = 1
except IOError:
    from_file = 0
    HSV_bounds = [[30, 20, 20], [90, 255, 255]]
    blur_amount = 1
    morph_amount = 1
    iterations = 1


def HSV_trackbar_name(P, bound):
    if P == 'H':
        P = 'Hue'
    if P == 'S':
        P = 'Saturation'
    if P == 'V':
        P = 'Value'
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
        if blur % 2 == 0:
            blur += 1
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
        if override_HSV_defaults or from_file:
            PD = Plant_Detection(image=filename,
                                 blur=blur, morph=morph, iterations=iterations,
                                 HSV_min=HSV_bounds[0], HSV_max=HSV_bounds[1],
                                 debug=True, save=False, text_output=False)
            PD.detect_plants()
            img = PD.final_debug_image
        else:
            PD = Plant_Detection(image=filename,
                                 blur=blur, morph=morph, iterations=iterations,
                                 debug=True, save=False, text_output=False)
            PD.detect_plants()
            img = PD.final_debug_image

        # Show processed image
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
    else:  # close window
        cv2.destroyWindow(HSVwindow)
        HSVwindow_loaded = 0

cv2.namedWindow(window)
cv2.createTrackbar('Blur', window, 0, 100, process)
cv2.createTrackbar('Morph', window, 1, 100, process)
cv2.createTrackbar('Iterations', window, 1, 100, process)
cv2.createTrackbar('Open HSV Selection Window', window, 0, 1, HSV_selection)

cv2.setTrackbarPos('Blur', window, blur_amount)
cv2.setTrackbarPos('Morph', window, morph_amount)
cv2.setTrackbarPos('Iterations', window, iterations)

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
