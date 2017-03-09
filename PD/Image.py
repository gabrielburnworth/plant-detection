#!/usr/bin/env python
"""Plant Detection Image Processing.

For Plant Detection.
"""
import sys
import os
import numpy as np
import cv2
try:
    from .Capture import Capture
except:  # noqa pylint:disable=W0702
    from Capture import Capture


class Image(object):
    """Provide image processes to Plant Detection."""

    def __init__(self, parameters, plant_db):
        """Set initial attributes.

        Arguments:
            Parameters() instance
            Database() instance
        """
        self.images = {
            'current': None, 'original': None, 'output': None, 'marked': None,
            'blurred': None, 'contoured': None, 'greyed': None,
            'morphed': None, 'morphed2': None, 'masked': None, 'masked2': None}
        self.output_text = True
        self.reduce_large = True
        self.params = parameters
        self.plant_db = plant_db
        self.object_count = None
        self.debug = False
        self.calibration_debug = False
        self.image_name = None
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.status = {'image': False, 'blur': False, 'mask': False,
                       'morph': False, 'bust': False, 'grey': False,
                       'mark': False, 'annotate': False}

    def _reduce(self):
        height, width = self.images['original'].shape[:2]
        if height > 600:
            self.images['output'] = cv2.resize(
                self.images['original'],
                (int(width * 600 / height), 600),
                interpolation=cv2.INTER_AREA)
        else:
            self.images['output'] = self.images['original'].copy()

    def _prepare(self):
        self._reduce()
        self.images['current'] = self.images['output'].copy()
        self.images['marked'] = self.images['output'].copy()
        self.status['image'] = True

    def load(self, filename):
        """Load image from file."""
        self.images['original'] = cv2.imread(filename, 1)
        if self.images['original'] is None:
            print("ERROR: Incorrect image path ({}).".format(filename))
            sys.exit(0)
        self.image_name = os.path.splitext(os.path.basename(filename))[0]
        self._prepare()

    def capture(self):
        """Capture image from camera."""
        image_filename = Capture().capture()
        self.images['original'] = self.load(image_filename)

    def save(self, title, image=None):
        """Save image to file."""
        if image is None:
            image = self.images['current']
        if self.image_name is None:
            name = ''
        else:
            name = '{}_'.format(self.image_name)
        filename = '{}{}{}.jpg'.format(self.dir, name, title)
        cv2.imwrite(filename, image)
        cv2.imwrite('/tmp/images/{}{}.jpg'.format(name, title),
                    image)

    def save_annotated(self, title):
        """Save annotated image to file."""
        self.save(title, image=self.annotate())

    def show(self):
        """Show image."""
        cv2.imshow("image", self.images['current'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotate(self, rotationangle):
        """Rotate image number of degrees."""
        try:
            rows, cols, _ = self.images['current'].shape
        except ValueError:
            rows, cols = self.images['current'].shape
        mtrx = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)),
                                       rotationangle, 1)
        self.images['current'] = cv2.warpAffine(
            self.images['current'], mtrx, (cols, rows))

    def rotate_main_images(self, rotationangle):
        """Rotate relevant working images."""
        self.images['current'] = self.images['output']  # work on output image
        self.rotate(rotationangle)  # rotate according to angle
        # create rotated image copies
        self.images['output'] = self.images['current'].copy()
        self.images['marked'] = self.images['current'].copy()

        try:
            self.images['morphed'].shape  # pylint:disable=W0104
            self.images['current'] = self.images['morphed']  # workon morphed
            self.rotate(rotationangle)  # rotate according to angle
            self.images['morphed'] = self.images['current'].copy()  # morphed
        except AttributeError:
            pass
        self.images['current'] = self.images['output']

    def blur(self):
        """Blur image."""
        if self.params.parameters['blur'] % 2 == 0:
            self.params.parameters['blur'] += 1
        self.images['blurred'] = cv2.medianBlur(
            self.images['current'], self.params.parameters['blur'])
        self.images['current'] = self.images['blurred'].copy()
        self.status['blur'] = True

    def mask(self):
        """Create mask using HSV range from blurred image."""
        # Create HSV image
        hsv = cv2.cvtColor(self.images['blurred'], cv2.COLOR_BGR2HSV)
        # Select HSV color bounds for mask and create plant mask
        # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
        hsv_min = [self.params.parameters['H'][0],
                   self.params.parameters['S'][0],
                   self.params.parameters['V'][0]]
        hsv_max = [self.params.parameters['H'][1],
                   self.params.parameters['S'][1],
                   self.params.parameters['V'][1]]
        if hsv_min[0] > hsv_max[0]:
            hsv_btwn_min = [0, hsv_min[1], hsv_min[2]]
            hsv_btwn_max = [179, hsv_max[1], hsv_max[2]]
            mask_lower = cv2.inRange(
                hsv, np.array(hsv_btwn_min), np.array(hsv_max))
            mask_upper = cv2.inRange(
                hsv, np.array(hsv_min), np.array(hsv_btwn_max))
            self.images['masked'] = cv2.addWeighted(
                mask_lower, 1.0, mask_upper, 1.0, 0.0)
        else:
            self.images['masked'] = cv2.inRange(
                hsv, np.array(hsv_min), np.array(hsv_max))
        self.images['current'] = self.images['masked'].copy()
        self.status['mask'] = True

    def _mask2(self):
        """Show regions of original image selected by mask."""
        self.images['masked2'] = cv2.bitwise_and(
            self.images['output'], self.images['output'],
            mask=self.images['masked'])
        temp = self.images['current']
        self.images['current'] = self.images['masked2']
        self.save_annotated('masked2')
        self.images['current'] = temp

    def morph(self):
        """Process mask to try to make plants more coherent."""
        if self.params.parameters['morph'] == 0:
            self.params.parameters['morph'] = 1
        if self.params.parameters['iterations'] == 0:
            self.params.parameters['iterations'] = 1
        if self.params.array is None:
            # Single morphological transformation
            kernel_type = self.params.cv2_kt[self.params.kernel_type]
            kernel = cv2.getStructuringElement(
                kernel_type,
                (self.params.parameters['morph'],
                 self.params.parameters['morph']))
            morph_type = self.params.cv2_mt[self.params.morph_type]
            self.images['morphed'] = cv2.morphologyEx(
                self.images['masked'], morph_type, kernel,
                iterations=self.params.parameters['iterations'])
        else:
            # List of morphological transformations
            processes = self.params.array
            self.images['morphed'] = self.images['masked']
            for process in processes:
                morph_amount = process['size']
                kernel_type = self.params.cv2_kt[process['kernel']]
                morph_type = self.params.cv2_mt[process['type']]
                iterations = process['iters']
                kernel = cv2.getStructuringElement(
                    kernel_type,
                    (morph_amount, morph_amount))
                if morph_type == 'erode':
                    self.images['morphed'] = cv2.erode(
                        self.images['morphed'], kernel, iterations=iterations)
                elif morph_type == 'dilate':
                    self.images['morphed'] = cv2.dilate(
                        self.images['morphed'], kernel, iterations=iterations)
                else:
                    self.images['morphed'] = cv2.morphologyEx(
                        self.images['morphed'],
                        morph_type, kernel, iterations=iterations)
        self.images['current'] = self.images['morphed']
        self.status['morph'] = True

    def _morph2(self):
        """Show regions of original image selected by morph."""
        self.images['morphed2'] = cv2.bitwise_and(
            self.images['output'], self.images['output'],
            mask=self.images['morphed'])
        temp = self.images['current']
        self.images['current'] = self.images['morphed2']
        self.save_annotated('morphed2')
        self.images['current'] = temp

    def initial_processing(self):
        """Process image in preparation for detecting plants."""
        # Blur image to simplify and reduce noise.
        self.blur()
        if self.debug:
            self.save_annotated('blurred')
        if self.calibration_debug:
            self.show()

        # Create a mask using the color range parameters
        self.mask()
        if self.debug:
            self.save_annotated('masked')
            self._mask2()
        if self.calibration_debug:
            self.show()

        # Transform mask to try to make objects more coherent
        self.morph()
        if self.debug:
            self.save_annotated('morphed')
            self._morph2()
        if self.calibration_debug:
            self.show()

    def clump_buster(self):
        """Break up selected regions of morphed image into smaller regions.

        Currently this is done by splitting the regions into quarters.
        """
        clump_img = self.images['morphed'].copy()
        try:
            # find contours in openCV 2: return hierarchy is unused
            contours, _ = cv2.findContours(
                clump_img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # find contours in openCV 3: return image and hierarchy are unused
            _, contours, _ = cv2.findContours(
                clump_img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(cnt)
            cv2.line(self.images['morphed'],
                     (int(rect_x + rect_w / 2.), rect_y),
                     (int(rect_x + rect_w / 2.), rect_y + rect_h),
                     (0), int(rect_w / 7.))
            cv2.line(self.images['morphed'],
                     (rect_x, int(rect_y + rect_h / 2.)),
                     (rect_x + rect_w, int(rect_y + rect_h / 2.)),
                     (0), int(rect_h / 7.))
        kernel = cv2.getStructuringElement(self.params.cv2_kt['ellipse'],
                                           (self.params.parameters['morph'],
                                            self.params.parameters['morph']))
        self.images['morphed'] = cv2.dilate(
            self.images['morphed'], kernel, iterations=1)
        self.images['current'] = self.images['morphed']
        self.status['bust'] = True

    def grey(self):
        """Grey out region in output image not selected by morphed mask."""
        grey_bg = cv2.addWeighted(np.full_like(self.images['output'], 255),
                                  0.4, self.images['output'], 0.6, 0)
        black_fg = cv2.bitwise_and(
            grey_bg, grey_bg, mask=cv2.bitwise_not(self.images['morphed']))
        plant_fg = cv2.bitwise_and(
            self.images['output'], self.images['output'],
            mask=self.images['morphed'])
        plant_fg_grey_bg = cv2.add(plant_fg, black_fg)
        self.images['greyed'] = plant_fg_grey_bg.copy()
        self.images['output'] = self.images['greyed']
        self.images['marked'] = self.images['greyed']
        self.status['grey'] = True

    def _find_contours(self):
        # Find contours (hopefully of outside edges of plants)
        contoured = self.images['morphed'].copy()
        try:
            # find contours in openCV 2: return hierarchy is unused
            contours, _ = cv2.findContours(
                contoured,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # find contours in openCV 3: return image and hierarchy are unused
            _, contours, _ = cv2.findContours(
                contoured,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            contoured = np.zeros_like(contoured, np.uint8)
        self.images['contoured'] = contoured
        return contours

    def _draw_contour(self, contour, calibration, draw_contours):
        # Draw contour
        if calibration and draw_contours:
            cv2.drawContours(
                self.images['marked'], [contour], 0, (0, 255, 0), 3)
        elif draw_contours:
            cv2.drawContours(
                self.images['contoured'], [contour], 0, (255, 255, 255), 3)
            cv2.drawContours(
                self.images['marked'], [contour], 0, (0, 0, 0), 6)
            cv2.drawContours(
                self.images['marked'], [contour], 0, (255, 255, 255), 2)

    def _save_calibration_contour(self, i, only_one_object, location=None):
        if i == 0:
            self.plant_db.calibration_pixel_locations = location
            only_one_object = True
        else:
            try:
                self.plant_db.calibration_pixel_locations = np.vstack(
                    (self.plant_db.calibration_pixel_locations,
                     location))
                only_one_object = False
            except ValueError:
                self.plant_db.calibration_pixel_locations = location
                only_one_object = True
        return only_one_object

    def find(self, calibration=False, safe_remove=False,
             draw_contours=True):
        """Create contours, find locations of objects, and mark them.

        Requires morphed image.
        """
        # Loop through contours
        contours = self._find_contours()
        if not safe_remove:
            self.plant_db.object_count = len(contours)
        self.plant_db.pixel_locations = []
        only_one_object = False
        for i, cnt in enumerate(contours):
            # Calculate plant location by using centroid of contour
            moment = cv2.moments(cnt)
            try:
                cnt_center_x = int(moment['m10'] / moment['m00'])
                cnt_center_y = int(moment['m01'] / moment['m00'])
                (cir_center_x,
                 cir_center_y), radius = cv2.minEnclosingCircle(cnt)
            except ZeroDivisionError:
                continue

            if calibration:
                # Mark calibration object with blue circle
                center = (int(cir_center_x), int(cir_center_y))
                cv2.circle(
                    self.images['marked'], center, int(radius), (255, 0, 0), 4)

            self._draw_contour(cnt, calibration, draw_contours)

            self.plant_db.pixel_locations.append(
                [cnt_center_x, cnt_center_y, radius])
            if calibration:
                only_one_object = self._save_calibration_contour(
                    i, only_one_object,
                    location=[cir_center_x, cir_center_y, radius])

        if calibration and only_one_object:
            self.plant_db.calibration_pixel_locations = [
                self.plant_db.calibration_pixel_locations]
            self.plant_db.object_count = 1
        if not safe_remove:
            self.images['current'] = self.images['contoured']
            self.status['mark'] = True

    def safe_remove(self, p2c):
        """Process plants marked as 'safe_remove'.

        Reprocess image to detect only the part of the plant
        outside of the known plant safe zone.
        """
        # Start with blank mask (black)
        safe_remove_img = np.zeros_like(self.images['morphed'], np.uint8)
        # Add safe-remove plants as (white) filled circles
        for plant in self.plant_db.plants['safe_remove']:
            self.plant_db.coordinate_locations = [
                plant['x'],
                plant['y'],
                plant['radius']]
            p2c.c2p(self.plant_db)
            point = np.array(self.plant_db.pixel_locations)[0]
            cv2.circle(safe_remove_img, (int(point[0]), int(point[1])),
                       int(point[2]), (255, 255, 255), -1)
        # Show only safe-remove plant shapes in original morphed mask
        # by applying safe-remove plant selection mask
        self.images['morphed'] = cv2.bitwise_and(
            self.images['morphed'], self.images['morphed'],
            mask=safe_remove_img)
        # Remove known plants and their safe zones from the mask
        for plant in self.plant_db.plants['known']:
            self.plant_db.coordinate_locations = [
                plant['x'],
                plant['y'],
                plant['radius']
                + self.plant_db.weeder_destrut_r]
            p2c.c2p(self.plant_db)
            point = np.array(self.plant_db.pixel_locations)[0]
            cv2.circle(self.images['morphed'], (int(point[0]), int(point[1])),
                       int(point[2]), (0, 0, 0), -1)
        # Detect the locations of the remaining plants in the mask
        self.find(safe_remove=True)
        p2c.p2c(self.plant_db)
        if not self.plant_db.plants['remove']:
            self.plant_db.plants['remove'] = []
        # The remaining plants (if any) should be weeds that can be
        # safely removed, but check again against known plants
        self.plant_db.identify(second_pass=True)

    def coordinates(self, p2c, draw_contours=True):
        """Detect coordinates of objects in image.

        Rotate image according to calibration data, detect objects and
        their coordinates.
        """
        # rotate according to calibration
        self.rotate_main_images(p2c.calibration_params['total_rotation_angle'])
        # set working image
        self.images['current'] = self.images['morphed']
        # detect pixel locations of objects
        self.find(draw_contours=draw_contours)
        # convert pixel locations to coordinates
        p2c.p2c(self.plant_db)

    def label(self, p2c=None, weeder_remove=False, weeder_safe_remove=False):
        """Draw circles on image indicating detected plants."""
        def _circle(color):
            bgr = {'red': (0, 0, 255),
                   'green': (0, 255, 0),
                   'blue': (255, 0, 0),
                   'cyan': (255, 255, 0),
                   'grey': (200, 200, 200)}
            for obj in self.plant_db.pixel_locations:
                cv2.circle(self.images['marked'], (int(obj[0]), int(obj[1])),
                           int(obj[2]), bgr[color], 4)

        if p2c is None:
            _circle('red')
        else:
            # Mark known plants
            known = [[_['x'], _['y'], _['radius']] for _
                     in self.plant_db.plants['known']]
            self.plant_db.coordinate_locations = known
            p2c.c2p(self.plant_db)
            _circle('green')

            # Mark weeds
            remove = [[_['x'], _['y'], _['radius']] for _
                      in self.plant_db.plants['remove']]
            self.plant_db.coordinate_locations = remove
            p2c.c2p(self.plant_db)
            _circle('red')

            # Mark weeder size for weeds
            if weeder_remove:
                weeder_size = self.plant_db.weeder_destrut_r
                remove_circle = [[_['x'], _['y'], weeder_size] for _
                                 in self.plant_db.plants['remove']]
                self.plant_db.coordinate_locations = remove_circle
                p2c.c2p(self.plant_db)
                _circle('grey')

            # Mark saved plants
            save = [[_['x'], _['y'], _['radius']] for _
                    in self.plant_db.plants['save']]
            self.plant_db.coordinate_locations = save
            p2c.c2p(self.plant_db)
            _circle('blue')

            # Mark safe-remove weeds
            safe_remove = [[_['x'], _['y'], _['radius']] for _
                           in self.plant_db.plants['safe_remove']]
            self.plant_db.coordinate_locations = safe_remove
            p2c.c2p(self.plant_db)
            _circle('cyan')

            # Mark weeder size for safe-remove weeds
            if weeder_safe_remove:
                weeder_size = self.plant_db.weeder_destrut_r
                safe_remove_circle = [[_['x'], _['y'], weeder_size] for _
                                      in self.plant_db.plants['safe_remove']]
                self.plant_db.coordinate_locations = safe_remove_circle
                p2c.c2p(self.plant_db)
                _circle('grey')

    def grid(self, p2c):
        """Draw grid on image indicating coordinate system."""
        width = self.images['marked'].shape[1]
        textsize = width / 2000.
        textweight = int(3.5 * textsize)

        def _grid_point(point, pointtype):
            if pointtype == 'coordinates':
                if len(point) < 3:
                    point = list(point) + [0]
                self.plant_db.coordinate_locations = point
                p2c.c2p(self.plant_db)
                point_pixel_location = np.array(
                    self.plant_db.pixel_locations)[0]
                pt_x = point_pixel_location[0]
                pt_y = point_pixel_location[1]
            else:  # pixels
                pt_x = point[0]
                pt_y = point[1]
            tps = width / 300.
            # crosshair center
            self.images['marked'][
                int(pt_y - tps):int(pt_y + tps + 1),
                int(pt_x - tps):int(pt_x + tps + 1)
            ] = (255, 255, 255)
            # crosshair lines
            self.images['marked'][
                int(pt_y - tps * 4):int(pt_y + tps * 4 + 1),
                int(pt_x - tps / 4):int(pt_x + tps / 4 + 1)
            ] = (255, 255, 255)
            self.images['marked'][
                int(pt_y - tps / 4):int(pt_y + tps / 4 + 1),
                int(pt_x - tps * 4):int(pt_x + tps * 4 + 1)
            ] = (255, 255, 255)
        # _grid_point([1650, 2050, 0], 'coordinates')  # test point
        _grid_point(Capture().getcoordinates(), 'coordinates')  # UTM location
        _grid_point(p2c.calibration_params['center_pixel_location'],
                    'pixels')  # image center

        grid_range = np.array([[x] for x in range(-10000, 10000, 100)])
        large_grid = np.hstack((grid_range, grid_range, grid_range))
        self.plant_db.coordinate_locations = large_grid
        p2c.c2p(self.plant_db)
        large_grid_pixel = np.array(self.plant_db.pixel_locations)
        for pixel_x, coord_x in zip(large_grid_pixel[:, 0], large_grid[:, 0]):
            if pixel_x > self.images['marked'].shape[1] or pixel_x < 0:
                continue
            self.images['marked'][
                :, int(pixel_x):int(pixel_x + 1)] = (255, 255, 255)
            cv2.putText(
                self.images['marked'], str(coord_x), (int(pixel_x), 100),
                cv2.FONT_HERSHEY_SIMPLEX, textsize,
                (255, 255, 255), textweight)
        for pixel_y, coord_y in zip(large_grid_pixel[:, 1], large_grid[:, 1]):
            if pixel_y > self.images['marked'].shape[0] or pixel_y < 0:
                continue
            self.images['marked'][
                int(pixel_y):int(pixel_y + 1), :] = (255, 255, 255)
            cv2.putText(
                self.images['marked'], str(coord_y), (100, int(pixel_y)),
                cv2.FONT_HERSHEY_SIMPLEX, textsize,
                (255, 255, 255), textweight)
        self.images['current'] = self.images['marked']

    def _add_annotation_text(self, lines):
        color_bgr = {'white': (255, 255, 255), 'black': (0, 0, 0)}
        color_value = {'white': 255, 'black': 0}
        textsize = self.images['current'].shape[1] / 1200.
        lineheight = int(40 * textsize)
        textweight = int(3.5 * textsize)
        font = cv2.FONT_HERSHEY_SIMPLEX
        add = lineheight + lineheight * len(lines)
        if self.status['morph'] and self.params.array:  # multiple morphs
            add_1 = add
            add += lineheight + lineheight * len(self.params.array)
        try:  # color image?
            color_array = self.images['current'].shape[2]
            new_shape = (self.images['current'].shape[0] + add,
                         self.images['current'].shape[1],
                         color_array)
        except IndexError:
            new_shape = (self.images['current'].shape[0] + add,
                         self.images['current'].shape[1])
        annotated_image = np.full(new_shape, color_value['black'], np.uint8)
        annotated_image[add:, :] = self.images['current']
        for line_num, line in enumerate(lines):
            cv2.putText(annotated_image, line,
                        (10, lineheight + line_num * lineheight),
                        font, textsize, color_bgr['white'], textweight)
        if self.status['morph'] and self.params.array:  # multiple morphs
            for line_num, line in enumerate(self.params.array):
                cv2.putText(annotated_image, str(line),
                            (10, add_1 + line_num * lineheight),
                            font, textsize, color_bgr['white'], textweight)
        return annotated_image

    def annotate(self):
        """Annotate image with processing parameters."""
        if self.status['blur']:
            lines = ["blur kernel size = {}".format(
                self.params.parameters['blur'])]
        else:
            return self.images['current']
        if self.status['mask']:
            hsv_min = [self.params.parameters['H'][0],
                       self.params.parameters['S'][0],
                       self.params.parameters['V'][0]]
            hsv_max = [self.params.parameters['H'][1],
                       self.params.parameters['S'][1],
                       self.params.parameters['V'][1]]
            lines = lines + [
                "HSV lower bound = {}".format(hsv_min),
                "HSV upper bound = {}".format(hsv_max)]
            if self.status['morph'] and not self.params.array:  # single morph
                lines = lines + [
                    "kernel type = {}".format(self.params.kernel_type),
                    "kernel size = {}".format(self.params.parameters['morph']),
                    "morphological transformation = {}".format(
                        self.params.morph_type),
                    "number of iterations = {}".format(
                        self.params.parameters['iterations'])]
        annotated_image = self._add_annotation_text(lines)
        self.status['annotate'] = True
        return annotated_image
