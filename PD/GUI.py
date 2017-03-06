#!/usr/bin/env python
"""Plant Detection GUI."""
import sys
import os
import json
import cv2
from Plant_Detection import Plant_Detection


class PlantDetectionGUI(object):
    """Interactively change input parameters.

    for Plant_Detection.detect_plants()
    """

    def __init__(self, image_filename=None):
        """Set initial attributes, get image path, and load inputs."""
        self.window = 'Plant Detection'
        self.hsv_window = 'HSV Selection'
        self.override_hsv_defaults = 0
        self.hsv_window_loaded = 0
        directory = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep

        # Image
        if image_filename:
            self.filename = image_filename
        else:
            if len(sys.argv) == 1:  # use test image
                self.filename = directory + 'soil_image.jpg'
            else:  # image filename provided in command line argument
                self.filename = sys.argv[1]

        # Load input parameters
        try:  # from file
            inputfilename = directory + "plant-detection_inputs.json"
            with open(inputfilename, 'r') as inputfile:
                inputs = json.load(inputfile)
            self.blur_amount = inputs['blur']
            self.morph_amount = inputs['morph']
            self.iterations = inputs['iterations']
            hsv_min = [inputs['H'][0], inputs['S'][0], inputs['V'][0]]
            hsv_max = [inputs['H'][1], inputs['S'][1], inputs['V'][1]]
            self.hsv_bounds = [hsv_min, hsv_max]
            self.from_file = 1
        except IOError:  # Use defaults
            self.from_file = 0
            self.hsv_bounds = [[30, 20, 20], [90, 255, 255]]
            self.blur_amount = 1
            self.morph_amount = 1
            self.iterations = 1

    @staticmethod
    def hsv_trackbar_name(parameter, bound):
        """Create GUI trackbar name."""
        if parameter == 'H':
            parameter = 'Hue'
        if parameter == 'S':
            parameter = 'Saturation'
        if parameter == 'V':
            parameter = 'Value'
        return '{} {} {}'.format(parameter, bound, ' ' * (12 - len(parameter)))

    def _get_hsv_values(self):
        # get HSV values from sliders in HSV window
        for bound_num, bound in enumerate(['min', 'max']):
            for parameter in range(0, 3):
                self.hsv_bounds[bound_num][parameter] = cv2.getTrackbarPos(
                    self.hsv_trackbar_name('HSV'[parameter], bound),
                    self.hsv_window)

    def process(self, _):
        """GUI trackbar callback."""
        hsv_window_open = cv2.getTrackbarPos(
            'Open HSV Selection Window', self.window)
        if hsv_window_open and not self.hsv_window_loaded:
            pass
        else:
            # Get parameter values
            blur = cv2.getTrackbarPos('Blur', self.window)
            morph = cv2.getTrackbarPos('Morph', self.window)
            iterations = cv2.getTrackbarPos('Iterations', self.window)

            if hsv_window_open:
                self._get_hsv_values()

            # Process image with parameters
            if self.override_hsv_defaults or self.from_file:
                plantdetection = Plant_Detection(
                    image=self.filename,
                    blur=blur, morph=morph,
                    iterations=iterations,
                    HSV_min=self.hsv_bounds[0],
                    HSV_max=self.hsv_bounds[1],
                    GUI=True)
                plantdetection.detect_plants()
                img = plantdetection.final_marked_image
            else:
                plantdetection = Plant_Detection(
                    image=self.filename,
                    blur=blur, morph=morph,
                    iterations=iterations,
                    GUI=True)
                plantdetection.detect_plants()
                img = plantdetection.final_marked_image

            # Show processed image
            cv2.imshow(self.window, img)

    def hsv_selection(self, open_window):
        """HSV selection GUI."""
        self.override_hsv_defaults = 1

        if open_window:
            cv2.namedWindow(self.hsv_window)
            for bound_num, bound in enumerate(['min', 'max']):
                for parameter, limit in zip(range(0, 3), [179, 255, 255]):
                    cv2.createTrackbar(
                        self.hsv_trackbar_name('HSV'[parameter], bound),
                        self.hsv_window, 0, limit, self.process)
                    cv2.setTrackbarPos(
                        self.hsv_trackbar_name('HSV'[parameter], bound),
                        self.hsv_window, self.hsv_bounds[bound_num][parameter])
            self.hsv_window_loaded = 1
        else:  # close window
            cv2.destroyWindow(self.hsv_window)
            self.hsv_window_loaded = 0

    def run(self):
        """Start the GUI."""
        cv2.namedWindow(self.window)
        cv2.createTrackbar('Blur', self.window, 0, 100, self.process)
        cv2.createTrackbar('Morph', self.window, 1, 100, self.process)
        cv2.createTrackbar('Iterations', self.window, 1, 100, self.process)
        cv2.createTrackbar('Open HSV Selection Window',
                           self.window, 0, 1, self.hsv_selection)

        cv2.setTrackbarPos('Blur', self.window, self.blur_amount)
        cv2.setTrackbarPos('Morph', self.window, self.morph_amount)
        cv2.setTrackbarPos('Iterations', self.window, self.iterations)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    GUI = PlantDetectionGUI()
    GUI.run()
