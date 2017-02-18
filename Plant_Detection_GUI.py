#!/usr/bin/env python
"""Plant Detection GUI."""
import sys
import os
import json
import cv2
from Plant_Detection import Plant_Detection


class Plant_Detection_GUI():
    """Interactively change input parameters for Plant_Detection.detect_plants()"""

    def __init__(self):
        directory = os.path.dirname(os.path.realpath(__file__)) + os.sep

        if len(sys.argv) == 1:
            self.filename = directory + 'soil_image.jpg'
        else:
            self.filename = sys.argv[1]
        self.window = 'Plant Detection'
        self.HSVwindow = 'HSV Selection'
        self.override_HSV_defaults = 0
        self.HSVwindow_loaded = 0
        try:  # Load input parameters from file
            with open(directory + "plant-detection_inputs.json", 'r') as f:
                inputs = json.load(f)
            self.blur_amount = inputs['blur']
            self.morph_amount = inputs['morph']
            self.iterations = inputs['iterations']
            HSV_min = [inputs['H'][0], inputs['S'][0], inputs['V'][0]]
            HSV_max = [inputs['H'][1], inputs['S'][1], inputs['V'][1]]
            self.HSV_bounds = [HSV_min, HSV_max]
            self.from_file = 1
        except IOError:
            self.from_file = 0
            self.HSV_bounds = [[30, 20, 20], [90, 255, 255]]
            self.blur_amount = 1
            self.morph_amount = 1
            self.iterations = 1

    def HSV_trackbar_name(self, P, bound):
        if P == 'H':
            P = 'Hue'
        if P == 'S':
            P = 'Saturation'
        if P == 'V':
            P = 'Value'
        return '{} {} {}'.format(P, bound, ' ' * (12 - len(P)))

    def process(self, _):
        HSVwindow_open = cv2.getTrackbarPos(
            'Open HSV Selection Window', self.window)
        if HSVwindow_open and not self.HSVwindow_loaded:
            pass
        else:
            # Get parameter values
            blur = cv2.getTrackbarPos('Blur', self.window)
            if blur % 2 == 0:
                blur += 1
            morph = cv2.getTrackbarPos('Morph', self.window)
            iterations = cv2.getTrackbarPos('Iterations', self.window)

            if HSVwindow_open:
                # get HSV values
                for b, bound in enumerate(['min', 'max']):
                    for P in range(0, 3):
                        self.HSV_bounds[b][P] = cv2.getTrackbarPos(
                            self.HSV_trackbar_name('HSV'[P], bound),
                            self.HSVwindow)

            # Process image with parameters
            if self.override_HSV_defaults or self.from_file:
                PD = Plant_Detection(image=self.filename,
                                     blur=blur, morph=morph, iterations=iterations,
                                     HSV_min=self.HSV_bounds[0],
                                     HSV_max=self.HSV_bounds[1],
                                     GUI=True)
                PD.detect_plants()
                img = PD.final_marked_image
            else:
                PD = Plant_Detection(image=self.filename,
                                     blur=blur, morph=morph, iterations=iterations,
                                     GUI=True)
                PD.detect_plants()
                img = PD.final_marked_image

            # Show processed image
            cv2.imshow(self.window, img)

    def HSV_selection(self, open_window):
        self.override_HSV_defaults = 1

        if open_window:
            cv2.namedWindow(self.HSVwindow)
            for b, bound in enumerate(['min', 'max']):
                for P, limit in zip(range(0, 3), [179, 255, 255]):
                    cv2.createTrackbar(
                        self.HSV_trackbar_name('HSV'[P], bound),
                        self.HSVwindow, 0, limit, self.process)
                    cv2.setTrackbarPos(
                        self.HSV_trackbar_name('HSV'[P], bound),
                        self.HSVwindow, self.HSV_bounds[b][P])
            self.HSVwindow_loaded = 1
        else:  # close window
            cv2.destroyWindow(self.HSVwindow)
            self.HSVwindow_loaded = 0

    def run(self):
        cv2.namedWindow(self.window)
        cv2.createTrackbar('Blur', self.window, 0, 100, self.process)
        cv2.createTrackbar('Morph', self.window, 1, 100, self.process)
        cv2.createTrackbar('Iterations', self.window, 1, 100, self.process)
        cv2.createTrackbar('Open HSV Selection Window',
                           self.window, 0, 1, self.HSV_selection)

        cv2.setTrackbarPos('Blur', self.window,  self.blur_amount)
        cv2.setTrackbarPos('Morph', self.window, self.morph_amount)
        cv2.setTrackbarPos('Iterations', self.window, self.iterations)

        while(1):
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    GUI = Plant_Detection_GUI()
    GUI.run()
