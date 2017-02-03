#!/usr/bin/env python
"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import sys, os
import numpy as np
import cv2
from PD.P2C import Pixel2coord
from PD.CeleryPy import FarmBotJSON
from PD.Image import Image
from PD.Parameters import Parameters
from PD.DB import DB

class Plant_Detection():
    """Detect plants in image and saves an image with plants marked.

       Kwargs:
           image (str): filename of image to process (default = None)
               None -> take photo instead
           coordinates (boolean): use coordinate conversion (default = False)
           calibration_img (filename): calibration image filename used to
               output coordinates instead of pixel locations (default = None)
           known_plants (list): [x, y, radius] of known (intentional) plants
                                (default = None)
           debug (boolean): output debug images (default = False)
           blur (int): blur kernel size (must be odd, default = 5)
           morph (int): amount of filtering (default = 5)
           iterations (int): number of morphological iterations (default = 1)
           array (list): list of morphs to run
               [[morph kernel size, morph kernel type, morph type, iterations]]
               example: array=[[3, 'cross', 'dilate', 2],
                               [5, 'rect',  'erode',  1]]
           save (boolean): save images (default = True)
           clump_buster (boolean): attempt to break
                                   plant clusters (default = False)
           HSV_min (list): green lower bound Hue(0-179), Saturation(0-255),
                           and Value(0-255) (default = [30, 20, 20])
           HSV_max (list): green upper bound Hue(0-179), Saturation(0-255),
                           and Value(0-255) (default = [90, 255, 255])
           parameters_from_file (boolean): load parameters from file
                                           (default = False)

       Examples:
           Detect_plants()
           Detect_plants(image='soil_image.jpg', morph=3, iterations=10,
              debug=True)
           Detect_plants(image='soil_image.jpg', blur=9, morph=7, iterations=4,
              calibration_img="PD/p2c_test_calibration.jpg")
           Detect_plants(image='soil_image.jpg', blur=15,
              array=[[5, 'ellipse', 'erode',  2],
                     [3, 'ellipse', 'dilate', 8]], debug=True, save=False,
              clump_buster=True, HSV_min=[15, 15, 15], HSV_max=[85, 245, 245])
    """
    def __init__(self, **kwargs):
        self.image = None
        self.coordinates = False
        self.calibration_img = None  # default
        self.known_plants = None  # default
        self.debug = False  # default
        self.save = True   # default
        self.parameters_from_file = False  # default
        self.params = Parameters()
        self.db = DB()
        for key in kwargs:
            if key == 'image': self.image = kwargs[key]
            if key == 'coordinates': self.coordinates = kwargs[key]
            if key == 'calibration_img': self.calibration_img = kwargs[key]
            if key == 'known_plants': self.known_plants = kwargs[key]
            if key == 'debug': self.debug = kwargs[key]
            if key == 'blur': self.params.blur_amount = kwargs[key]
            if key == 'morph': self.params.morph_amount = kwargs[key]
            if key == 'iterations': self.params.iterations = kwargs[key]
            if key == 'array': self.params.array = kwargs[key]
            if key == 'save': self.save = kwargs[key]
            if key == 'clump_buster': self.params.clump_buster = kwargs[key]
            if key == 'HSV_min': self.params.HSV_min = kwargs[key]
            if key == 'HSV_max': self.params.HSV_max = kwargs[key]
            if key == 'parameters_from_file':
                self.parameters_from_file = kwargs[key]
        if self.calibration_img is not None:
            self.coordinates = True
        self.grey_out = False
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.output_text = True
        self.output_json = False
        self.input_parameters_file = self.dir + "plant-detection_inputs.txt"
        self.known_plants_file = self.dir + "plant-detection_known-plants.txt"

    def calibrate(self):
        if self.calibration_img is None and self.coordinates:
            self.calibration_img = self._getimage()
        P2C = Pixel2coord(calibration_image=self.calibration_img)

    def detect_plants(self):
        if self.parameters_from_file:
            self.params.load(self.input_parameters_file)

        if self.output_text:
            print("\nProcessing image: {}".format(self.image))
        kt = None; upper_green = None

        # Load image
        if self.image is None:
            self.image = Image(self.params, self.db)
            self.image.capture()
            self.image.save('photo')
        else:
            filename = self.image
            self.image = Image(self.params, self.db)
            self.image.load(filename)

        self.image.blur()
        if self.debug:
            self.image.save('blurred')

        self.image.mask()
        if self.debug:
            self.image.save('masked')
            self.image.mask2()

        self.image.morph()
        if self.debug:
            self.image.save('morphed')
            self.image.morph2()

        if self.params.clump_buster:
            self.image.clump_buster()

        if self.grey_out:
            self.image.grey()

        if not self.coordinates:
            self.db.pixel_locations = self.image.find()
            self.image.save('contours')
            if self.output_text:
                self.db.print_pixel()
            # Save soil image with plants marked
            self.image.image = self.image.marked
            self.image.save('marked')

        # Return coordinates if requested
        if self.coordinates:
            P2C = Pixel2coord()

            def rotateimage(image, rotationangle):
                try:
                    rows, cols, _ = image.shape
                except ValueError:
                    rows, cols = image.shape
                mtrx = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)),
                                               rotationangle, 1)
                return cv2.warpAffine(image, mtrx, (cols, rows))
            inputimage = rotateimage(self.image.original, P2C.total_rotation_angle)
            self.image.morphed = rotateimage(self.image.morphed,
                                             P2C.total_rotation_angle)
            self.image.image = self.image.morphed
            self.db.pixel_locations = self.image.find()
            P2C.test_coordinates = [2000, 2000]
            plant_coordinates, plant_pixel_locations = P2C.p2c(
                self.db.pixel_locations)

            if self.debug:
                self.image.save('contours')
                self.image.image = self.image.marked
                self.image.save('coordinates_found')
            self.image.marked = self.image.original.copy()

            # Find unknown
            marked, unmarked = [], []
            if self.known_plants is None:
                known_plants = [[0, 0, 0]]
            else:
                known_plants = self.known_plants
            kplants = np.array(known_plants)
            for plant_coord in plant_coordinates:
                x, y, r = plant_coord[0], plant_coord[1], plant_coord[2]
                cxs, cys, crs = kplants[:, 0], kplants[:, 1], kplants[:, 2]
                if all((x - cx)**2 + (y - cy)**2 > cr**2
                       for cx, cy, cr in zip(cxs, cys, crs)):
                    marked.append([x, y, r])
                else:
                    unmarked.append([x, y, r])


            self.db.marked = marked
            self.db.unmarked = unmarked
            if self.output_text:
                self.db.print_()
            if self.output_json:
                self.db.json_()

            # Create annotated image
            known_PL = P2C.c2p(known_plants)
            marked_PL = P2C.c2p(marked)
            unmarked_PL = P2C.c2p(unmarked)
            for mark in marked_PL:
                cv2.circle(self.image.marked, (int(mark[0]), int(mark[1])),
                           int(mark[2]), (0, 0, 255), 4)
            for known in known_PL:
                cv2.circle(self.image.marked, (int(known[0]), int(known[1])),
                           int(known[2]), (0, 255, 0), 4)
            for unmarked in unmarked_PL:
                cv2.circle(self.image.marked, (int(unmarked[0]),
                                        int(unmarked[1])),
                           int(unmarked[2]), (255, 0, 0), 4)
            if 0:
                for ppl in plant_pixel_locations[1:]:
                    cv2.circle(self.image.marked, (int(ppl[0]), int(ppl[1])),
                               int(ppl[2]), (0, 0, 0), 4)

            # Grid
            w = self.image.marked.shape[1]
            textsize = w / 2000.
            textweight = int(3.5 * textsize)
            def grid_point(point, pointtype):
                if pointtype == 'coordinates':
                    if len(point) < 3:
                        point = list(point) + [0]
                    point_pixel_location = np.array(P2C.c2p(point))[0]
                    x = point_pixel_location[0]
                    y = point_pixel_location[1]
                else:  # pixels
                    x = point[0]
                    y = point[1]
                tps = w / 300.
                # crosshair center
                self.image.marked[int(y - tps):int(y + tps + 1),
                           int(x - tps):int(x + tps + 1)] = (255, 255, 255)
                # crosshair lines
                self.image.marked[int(y - tps * 4):int(y + tps * 4 + 1),
                           int(x - tps / 4):int(x + tps / 4 + 1)] = (255,
                                                                     255,
                                                                     255)
                self.image.marked[int(y - tps / 4):int(y + tps / 4 + 1),
                           int(x - tps * 4):int(x + tps * 4 + 1)] = (255,
                                                                     255,
                                                                     255)
            #grid_point([1650, 2050, 0], 'coordinates')  # test point
            grid_point(P2C.test_coordinates, 'coordinates')  # UTM location
            grid_point(P2C.center_pixel_location, 'pixels')  # image center

            grid_range = np.array([[x] for x in range(0, 20000, 100)])
            large_grid = np.hstack((grid_range, grid_range, grid_range))
            large_grid_pl = np.array(P2C.c2p(large_grid))
            for x, xc in zip(large_grid_pl[:, 0], large_grid[:, 0]):
                if x > self.image.marked.shape[1] or x < 0:
                    continue
                self.image.marked[:, int(x):int(x + 1)] = (255, 255, 255)
                cv2.putText(self.image.marked, str(xc), (int(x), 100),
                            cv2.FONT_HERSHEY_SIMPLEX, textsize,
                            (255, 255, 255), textweight)
            for y, yc in zip(large_grid_pl[:, 1], large_grid[:, 1]):
                if y > self.image.marked.shape[0] or y < 0:
                    continue
                self.image.marked[int(y):int(y + 1), :] = (255, 255, 255)
                cv2.putText(self.image.marked, str(yc), (100, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, textsize,
                            (255, 255, 255), textweight)
            self.image.image = self.image.marked
            self.image.save('marked')

        if self.debug:
            self.final_debug_image = self.image.marked
            self.params.save(self.input_parameters_file)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        soil_image = dir + 'soil_image.jpg'
        PD = Plant_Detection(image=soil_image,
            blur=15, morph=6, iterations=4,
            calibration_img=dir + "PD/p2c_test_calibration.jpg", debug=1,
            known_plants=[[1600, 2200, 100], [2050, 2650, 120]])
        PD.calibrate()
        PD.detect_plants()
    else:
        soil_image = sys.argv[1]
        PD = Plant_Detection(image=soil_image, parameters_from_file=True, debug=True)
        PD.detect_plants()
