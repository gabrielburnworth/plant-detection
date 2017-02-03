#!/usr/bin/env python
"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import sys, os
import numpy as np
import cv2
import platform
import json
use_rpi_camera = False; using_rpi = False
if platform.uname()[4].startswith("arm") and use_rpi_camera:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    using_rpi = True
from time import sleep
from pixel_to_coordinate.pixel2coord import Pixel2coord
from CeleryPy import FarmBotJSON


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
              calibration_img="pixel_to_coordinate/p2c_test_calibration.jpg")
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
        self.blur_amount = 5  # default
        self.morph_amount = 5  # default
        self.iterations = 1  # default
        self.array = None  # default
        self.save = True   # default
        self.clump_buster = False  # default
        self.HSV_min = [30, 20, 20]  # default
        self.HSV_max = [90, 255, 255]  # default
        self.parameters_from_file = False  # default
        for key in kwargs:
            if key == 'image': self.image = kwargs[key]
            if key == 'coordinates': self.coordinates = kwargs[key]
            if key == 'calibration_img': self.calibration_img = kwargs[key]
            if key == 'known_plants': self.known_plants = kwargs[key]
            if key == 'debug': self.debug = kwargs[key]
            if key == 'blur': self.blur_amount = kwargs[key]
            if key == 'morph': self.morph_amount = kwargs[key]
            if key == 'iterations': self.iterations = kwargs[key]
            if key == 'array': self.array = kwargs[key]
            if key == 'save': self.save = kwargs[key]
            if key == 'clump_buster': self.clump_buster = kwargs[key]
            if key == 'HSV_min': self.HSV_min = kwargs[key]
            if key == 'HSV_max': self.HSV_max = kwargs[key]
            if key == 'parameters_from_file':
                self.parameters_from_file = kwargs[key]
        if self.calibration_img is not None:
            self.coordinates = True
        self.test_coordinates = [2000, 2000]
        self.grey_out = False
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.output_text = True
        self.output_json = False
        self.input_parameters_filename = "plant-detection_inputs.txt"
        self.input_parameters_file = self.dir + self.input_parameters_filename
        self.known_plants_filename = "plant-detection_known-plants.txt"
        self.known_plants_file = self.dir + self.known_plants_filename
        self.tmp_dir = None

        try:
            params_json = json.loads(os.environ['PLANT_DETECTION.options'])
            # Read inputs from env vars
            self.HSV_min = [params_json['H'][0], params_json['S'][0], params_json['V'][0]]
            self.HSV_max = [params_json['H'][1], params_json['S'][1], params_json['V'][1]]
            self.blur_amount = int(params_json['blur'])
            self.morph_amount = int(params_json['morph'])
            self.iterations = int(params_json['iterations'])
        except KeyError:
            pass

    def _getcoordinates(self):
        """Get machine coordinates from bot."""
        # For now, return testing coordintes:
        return self.test_coordinates

    def _getimage(self):
        """Take a photo."""
        if using_rpi and use_rpi_camera:
            # With Raspberry Pi Camera:
            with PiCamera() as camera:
                camera.resolution = (1920, 1088)
                rawCapture = PiRGBArray(camera)
                sleep(0.1)
                camera.capture(rawCapture, format="bgr")
                image = rawCapture.array
        else:
            # With USB cameras:
            camera = cv2.VideoCapture(0)
            sleep(0.1)
            _, image = camera.read()
            camera.release()
        self.current_coordinates = self._getcoordinates()
        filename = '{}_{}.png'.format(*self.current_coordinates)
        cv2.imwrite(filename, image)
        cv2.imwrite('/tmp/images/' + filename, image)
        if self.output_text:
            print("Image saved: {}".format(filename))
        return image

    def calibrate(self):
        if self.calibration_img is None and self.coordinates:
            self.calibration_img = self._getimage()
        P2C = Pixel2coord(calibration_image=self.calibration_img)

    def detect_plants(self):

        def save_detected_plants(save, remove):
            if self.tmp_dir is None:
                csv_dir = self.dir
            else:
                csv_dir = self.tmp_dir
            try:
                np.savetxt(csv_dir + "detected-plants_saved.csv", save,
                           fmt='%.1f', delimiter=',', header='X,Y,Radius')
                np.savetxt(csv_dir + "detected-plants_to-remove.csv", remove,
                           fmt='%.1f', delimiter=',', header='X,Y,Radius')
            except IOError:
                self.tmp_dir = "/tmp/"
                save_detected_plants(save, remove)

        def save_parameters():
            if self.tmp_dir is None:
                filename = self.input_parameters_file
            else:
                filename = self.tmp_dir + self.input_parameters_filename
            with open(filename, 'w') as f:
                f.write('blur_amount {}\n'.format(self.blur_amount))
                f.write('morph_amount {}\n'.format(self.morph_amount))
                f.write('iterations {}\n'.format(self.iterations))
                f.write('clump_buster {}\n'.format(
                    [1 if self.clump_buster else 0][0]))
                f.write('HSV_min {:d} {:d} {:d}\n'.format(*self.HSV_min))
                f.write('HSV_max {:d} {:d} {:d}\n'.format(*self.HSV_max))
            if self.tmp_dir is None:
                filename = self.known_plants_file
            else:
                filename = self.tmp_dir + self.known_plants_filename
            with open(filename, 'w') as f:
                f.write('X Y Radius\n')
                if self.known_plants is not None:
                    for plant in self.known_plants:
                        f.write('{} {} {}\n'.format(*plant))

        def load_parameters():
            if self.tmp_dir is None:
                filename = self.input_parameters_file
            else:
                filename = self.tmp_dir + self.input_parameters_filename
            try:  # Load input parameters from file
                with open(filename, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    if "blur_amount" in line:
                        self.blur_amount = int(line[1])
                        if self.blur_amount % 2 == 0:
                            self.blur_amount += 1
                    if "morph_amount" in line:
                        self.morph_amount = int(line[1])
                    if "iterations" in line:
                        self.iterations = int(line[1])
                    if "clump_buster" in line:
                        self.clump_buster = int(line[1])
                    if "HSV_min" in line:
                        self.HSV_min = [int(line[1]),
                                        int(line[2]),
                                        int(line[3])]
                    if "HSV_max" in line:
                        self.HSV_max = [int(line[1]),
                                        int(line[2]),
                                        int(line[3])]
                if self.tmp_dir is None:
                    filename = self.known_plants_file
                else:
                    filename = self.tmp_dir + self.known_plants_filename
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    known_plants = []
                    for line in lines[1:]:
                        line = line.strip().split(' ')
                        known_plants.append([float(line[0]),
                                             float(line[1]),
                                             float(line[2])])
                    if len(known_plants) > 0:
                        self.known_plants = known_plants
            except IOError:  # Use defaults and save to file
                try:
                    save_parameters()
                except IOError:
                    self.tmp_dir = "/tmp/"
                    save_parameters()

        if self.parameters_from_file:
            load_parameters()

        def save_image(img, step, description):
            save_image = img
            if step is not None:  # debug image
                save_image = annotate(img)
            if isinstance(self.image, str):
                name = self.image[:-4]
            else:
                name = '{}_{}'.format(*self.test_coordinates)
            details = description
            if step is not None:  # debug image
                details = 'debug-{}_{}'.format(step, details)
            filename = '{}_{}.png'.format(name, details)
            if self.save:
                try:
                    cv2.imwrite(filename, save_image)
                except IOError:
                    pass
                cv2.imwrite('/tmp/images/image_' + details + '.jpg', save_image)
                if self.output_text:
                    print("Image saved: {}".format(filename))
            return save_image

        def annotate(img):
            font = cv2.FONT_HERSHEY_SIMPLEX
            lines = ["blur kernel size = {}".format(self.blur_amount)]  # blur
            if upper_green is not None:  # color mask
                lines = lines + [
                    "HSV green lower bound = {}".format(lower_green),
                    "HSV green upper bound = {}".format(upper_green)]
                if self.array is None and kt is not None:  # single morph
                    lines = lines + [
                        "kernel type = {}".format(kernel_type),
                        "kernel size = {}".format(self.morph_amount),
                        "morphological transformation = {}".format(morph_type),
                        "number of iterations = {}".format(self.iterations)]
            h = img.shape[0]; w = img.shape[1]
            textsize = w / 1200.
            lineheight = int(40 * textsize); textweight = int(3.5 * textsize)
            add = lineheight + lineheight * len(lines)
            if self.array is not None and kt is not None:  # multiple morphs
                add_1 = add
                add += lineheight + lineheight * len(self.array)
            try:  # color image?
                c = img.shape[2]
                new_shape = (h + add, w, c)
            except IndexError:
                new_shape = (h + add, w)
            annotated_image = np.zeros(new_shape, np.uint8)
            annotated_image[add:, :] = img
            for o, line in enumerate(lines):
                cv2.putText(annotated_image, line,
                            (10, lineheight + o * lineheight),
                            font, textsize, (255, 255, 255), textweight)
            if self.array is not None and kt is not None:  # multiple morphs
                for o, line in enumerate(array):
                    cv2.putText(annotated_image, str(line),
                                (10, add_1 + o * lineheight),
                                font, textsize, (255, 255, 255), textweight)
            return annotated_image

        if self.output_text:
            print("\nProcessing image: {}".format(self.image))
        kt = None; upper_green = None

        # Load image and create blurred image
        if self.image is None:
            self.image = self._getimage()
            original_image = self.image
            save_image(original_image, None, 'photo')
        else:
            original_image = cv2.imread(self.image, 1)
        height, width = original_image.shape[:2]
        if height > 600:
            original_image = cv2.resize(original_image,
                (int(width * 600 / height), 600), interpolation=cv2.INTER_AREA)
        img0 = original_image.copy()
        img = original_image.copy()
        blur = cv2.medianBlur(img, self.blur_amount)
        if self.debug:
            img2 = img.copy()
            save_image(blur, 0, 'blurred')

        # Create HSV image
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # Select HSV color bounds for mask and create plant mask
        # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
        lower_green = self.HSV_min
        upper_green = self.HSV_max
        if self.HSV_min[0] > self.HSV_max[0]:
            hsv_btwn_min = [0, self.HSV_min[1], self.HSV_min[2]]
            hsv_btwn_max = [179, self.HSV_max[1], self.HSV_max[2]]
            mask_L = cv2.inRange(hsv, np.array(hsv_btwn_min),
                                 np.array(self.HSV_max))
            mask_U = cv2.inRange(hsv, np.array(self.HSV_min),
                                 np.array(hsv_btwn_max))
            mask = cv2.addWeighted(mask_L, 1.0, mask_U, 1.0, 0.0)
        else:
            mask = cv2.inRange(hsv, np.array(self.HSV_min),
                               np.array(self.HSV_max))
        if self.debug:
            save_image(mask, 1, 'mask')
            res = cv2.bitwise_and(img, img, mask=mask)
            save_image(res, 2, 'masked')

        # Create dictionaries of morph types
        kt = {}  # morph kernel type
        kt['ellipse'] = cv2.MORPH_ELLIPSE
        kt['rect'] = cv2.MORPH_RECT
        kt['cross'] = cv2.MORPH_CROSS
        mt = {}  # morph type
        mt['close'] = cv2.MORPH_CLOSE
        mt['open'] = cv2.MORPH_OPEN

        # Process mask to try to make plants more coherent
        if self.array is None:
            # Single morphological transformation
            kernel_type = 'ellipse'
            kernel = cv2.getStructuringElement(kt[kernel_type],
                                               (self.morph_amount,
                                                self.morph_amount))
            morph_type = 'close'
            proc = cv2.morphologyEx(mask,
                                    mt[morph_type], kernel,
                                    iterations=self.iterations)
        else:
            # List of morphological transformations
            processes = self.array; self.array = None
            proc = mask
            for p, process in enumerate(processes):
                morph_amount = process[0]; kernel_type = process[1]
                morph_type = process[2]; iterations = process[3]
                kernel = cv2.getStructuringElement(kt[kernel_type],
                                                  (morph_amount, morph_amount))
                if morph_type == 'erode':
                    proc = cv2.erode(proc, kernel, iterations=iterations)
                elif morph_type == 'dilate':
                    proc = cv2.dilate(proc, kernel, iterations=iterations)
                else:
                    proc = cv2.morphologyEx(proc,
                                            mt[morph_type], kernel,
                                            iterations=iterations)
                save_image(proc, '3p{}'.format(p), 'processed-mask')
            self.array = processes
        if self.debug:
            save_image(proc, 4, 'processed-mask')
            res2 = cv2.bitwise_and(img, img, mask=proc)
            save_image(res2, 5, 'processed-masked')

        if self.clump_buster:
            cb_proc = proc.copy()
            try:
                contours, hierarchy = cv2.findContours(cb_proc,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                unused_img, contours, hierarchy = cv2.findContours(cb_proc,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours)):
                cnt = contours[i]
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                cv2.line(proc, (rx + rw / 2, ry), (rx + rw / 2, ry + rh),
                         (0), rw / 7)
                cv2.line(proc, (rx, ry + rh / 2), (rx + rw, ry + rh / 2),
                         (0), rh / 7)
            proc = cv2.dilate(proc, kernel, iterations=1)

        if self.grey_out:
            # Grey out region not selected by mask
            grey_bg = cv2.addWeighted(np.full_like(img, 255), 0.4, img, 0.6, 0)
            black_fg = cv2.bitwise_and(grey_bg, grey_bg, mask=cv2.bitwise_not(proc))
            plant_fg_grey_bg = cv2.add(cv2.bitwise_and(img, img, mask=proc), black_fg)
            img0 = plant_fg_grey_bg.copy()
            img = plant_fg_grey_bg.copy()
            img2 = plant_fg_grey_bg.copy()

        def find(proc):
            # Find contours (hopefully of outside edges of plants)
            try:
                contours, hierarchy = cv2.findContours(proc,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                unused_img, contours, hierarchy = cv2.findContours(proc,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            if self.output_text:
                print("{} plants detected in image.".format(len(contours)))

            # Loop through contours
            for i, cnt in enumerate(contours):
                # Calculate plant location by using centroid of contour
                M = cv2.moments(cnt)
                try:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    (_, _), radius = cv2.minEnclosingCircle(cnt)
                except ZeroDivisionError:
                    continue
                if not self.coordinates and self.output_text:
                    if i == 0:
                        print("Detected plant center pixel locations ( X Y ):")
                    print("    ( {:5.0f}px {:5.0f}px )".format(cx, cy))

                # Mark plant with red circle
                cv2.circle(img, (cx, cy), 20, (0, 0, 255), 4)

                if self.debug:
                    cv2.drawContours(proc, [cnt], 0, (255, 255, 255), 3)
                    cv2.circle(img2, (cx, cy), 20, (0, 0, 255), 4)
                    cv2.drawContours(img2, [cnt], 0, (0, 0, 0), 6)
                    cv2.drawContours(img2, [cnt], 0, (255, 255, 255), 2)

                object_pixel_locations.append([cx, cy, radius])
            return object_pixel_locations

        object_pixel_locations = []
        if not self.coordinates:
            object_pixel_locations = find(proc)

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
            inputimage = rotateimage(img0, P2C.total_rotation_angle)
            proc = rotateimage(proc, P2C.total_rotation_angle)
            object_pixel_locations = find(proc)
            P2C.test_coordinates = self._getcoordinates()
            plant_coordinates, plant_pixel_locations = P2C.p2c(
                object_pixel_locations)

            if self.debug: save_image(img2, None, 'coordinates_found')
            marked_img = inputimage.copy()

            # Known plant exclusion:
            if self.output_text:
                if self.known_plants is not None:
                    # Print known
                    print("\n{} known plants inputted.".format(
                        len(self.known_plants)))
                    if len(self.known_plants) > 0:
                        print("Plants at the following machine coordinates "
                              "( X Y ) with R = radius are to be saved:")
                    for known_plant in self.known_plants:
                        print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(
                            *known_plant))
                else:
                    print("\n No known plants inputted.")

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

            # Print removal candidates
            if self.output_text:
                print("\n{} plants marked for removal.".format(len(marked)))
                if len(marked) > 0:
                    print("Plants at the following machine coordinates "
                          "( X Y ) with R = radius are to be removed:")
                for mark in marked:
                    print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*mark))

            # Print saved
            if self.output_text:
                print("\n{} detected plants are known or have escaped "
                      "removal.".format(len(unmarked)))
                if len(unmarked) > 0:
                    print("Plants at the following machine coordinates "
                          "( X Y ) with R = radius have been saved:")
                for unmark in unmarked:
                    print("    ( {:5.0f} {:5.0f} ) R = {:.0f}".format(*unmark))

            # Encode to CS
            if self.output_json:
                FarmBot = FarmBotJSON()
                for mark in marked:
                    x, y = round(mark[0], 2), round(mark[1], 2)
                    FarmBot.add_point(x, y, 0)
                for unmark in unmarked:
                    x, y = round(unmark[0], 2), round(unmark[1], 2)
                    r = round(unmark[2], 2)
                    FarmBot.add_plant(0, [x, y, 0], r)

            # Save plant coordinates to file
            save_detected_plants(unmarked, marked)

            # Create annotated image
            known_PL = P2C.c2p(known_plants)
            marked_PL = P2C.c2p(marked)
            unmarked_PL = P2C.c2p(unmarked)
            for mark in marked_PL:
                cv2.circle(marked_img, (int(mark[0]), int(mark[1])),
                           int(mark[2]), (0, 0, 255), 4)
            for known in known_PL:
                cv2.circle(marked_img, (int(known[0]), int(known[1])),
                           int(known[2]), (0, 255, 0), 4)
            for unmarked in unmarked_PL:
                cv2.circle(marked_img, (int(unmarked[0]),
                                        int(unmarked[1])),
                           int(unmarked[2]), (255, 0, 0), 4)
            if 0:
                for ppl in plant_pixel_locations[1:]:
                    cv2.circle(marked_img, (int(ppl[0]), int(ppl[1])),
                               int(ppl[2]), (0, 0, 0), 4)

            # Grid
            w = marked_img.shape[1]
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
                marked_img[int(y - tps):int(y + tps + 1),
                           int(x - tps):int(x + tps + 1)] = (255, 255, 255)
                # crosshair lines
                marked_img[int(y - tps * 4):int(y + tps * 4 + 1),
                           int(x - tps / 4):int(x + tps / 4 + 1)] = (255,
                                                                     255,
                                                                     255)
                marked_img[int(y - tps / 4):int(y + tps / 4 + 1),
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
                if x > marked_img.shape[1] or x < 0:
                    continue
                marked_img[:, int(x):int(x + 1)] = (255, 255, 255)
                cv2.putText(marked_img, str(xc), (int(x), 100),
                            cv2.FONT_HERSHEY_SIMPLEX, textsize,
                            (255, 255, 255), textweight)
            for y, yc in zip(large_grid_pl[:, 1], large_grid[:, 1]):
                if y > marked_img.shape[0] or y < 0:
                    continue
                marked_img[int(y):int(y + 1), :] = (255, 255, 255)
                cv2.putText(marked_img, str(yc), (100, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, textsize,
                            (255, 255, 255), textweight)
            save_image(marked_img, None, 'marked')

        if self.debug:
            save_image(proc, 6, 'contours')
            self.final_debug_image = save_image(img2, 7, 'img-contours')
            save_parameters()

        # Save soil image with plants marked
        if not self.coordinates:
            save_image(img, None, 'marked')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        soil_image = dir + 'soil_image.jpg'
        PD = Plant_Detection(image=soil_image,
            blur=15, morph=6, iterations=4,
            calibration_img=dir + "pixel_to_coordinate/p2c_test_calibration.jpg",
            known_plants=[[1600, 2200, 100], [2050, 2650, 120]])
        PD.calibrate()
        PD.detect_plants()
    else:
        soil_image = sys.argv[1]
        PD = Plant_Detection(image=soil_image, parameters_from_file=True)
        PD.detect_plants()
