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
    from .Parameters import Parameters
    from .DB import DB
except:
    from Capture import Capture
    from Parameters import Parameters
    from DB import DB


class Image():
    """Provide image processes to Plant Detection"""

    def __init__(self, parameters, db):
        self.image = None  # working image
        self.original = None
        self.output = None
        self.blurred = None
        self.morphed = None
        self.morphed2 = None
        self.masked = None
        self.masked2 = None
        self.marked = None
        self.contoured = None
        self.output_text = True
        self.reduce_large = True
        self.greyed = None
        self.params = parameters
        self.db = db
        self.object_count = None
        self.debug = False
        self.calibration_debug = False
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.status = {'image': False, 'blur': False, 'mask': False,
                       'morph': False, 'bust': False, 'grey': False,
                       'mark': False, 'annotate': False}

    def _reduce(self):
        height, width = self.original.shape[:2]
        if height > 600:
            self.output = cv2.resize(self.original,
                                     (int(width * 600 / height), 600),
                                     interpolation=cv2.INTER_AREA)
        else:
            self.output = self.original.copy()

    def _prepare(self):
        self._reduce()
        self.image = self.output.copy()
        self.marked = self.output.copy()
        self.status['image'] = True

    def load(self, filename):
        """Load image from file"""
        self.original = cv2.imread(filename, 1)
        if self.original is None:
            print("ERROR: Incorrect image path ({}).".format(filename))
            sys.exit(0)
        self._prepare()

    def capture(self):
        """Capture image from camera"""
        self.original = Capture().capture()
        self._prepare()

    def save(self, title):
        """Save image to file"""
        filename = '{}{}.jpg'.format(self.dir, title)
        cv2.imwrite(filename, self.image)
        cv2.imwrite('/tmp/images/image_{}.jpg'.format(title), self.image)

    def save_annotated(self, title):
        """Save annotated image to file"""
        filename = '{}{}.jpg'.format(self.dir, title)
        cv2.imwrite(filename, self.annotate())
        cv2.imwrite('/tmp/images/image_{}.jpg'.format(title), self.annotate())

    def show(self):
        """Show image."""
        cv2.imshow("image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotate(self, rotationangle):
        """Rotate image number of degrees."""
        try:
            rows, cols, _ = self.image.shape
        except ValueError:
            rows, cols = self.image.shape
        mtrx = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)),
                                       rotationangle, 1)
        self.image = cv2.warpAffine(self.image, mtrx, (cols, rows))

    def rotate_main_images(self, rotationangle):
        """Rotate relevant working images"""
        self.image = self.output  # work on output image
        self.rotate(rotationangle)  # rotate according to angle
        self.output = self.image.copy()
        self.marked = self.image.copy()  # create copy of rotated img to mark up

        try:
            self.morphed.shape
            self.image = self.morphed  # work on morphed mask
            self.rotate(rotationangle)  # rotate according to angle
            self.morphed = self.image.copy()  # save to morphed mask
        except AttributeError:
            pass
        self.image = self.output

    def blur(self):
        """Blur image"""
        if self.params.parameters['blur'] % 2 == 0:
            self.params.parameters['blur'] += 1
        self.blurred = cv2.medianBlur(
            self.image, self.params.parameters['blur'])
        self.image = self.blurred.copy()
        self.status['blur'] = True

    def mask(self):
        """Create mask using HSV range from blurred image"""
        # Create HSV image
        hsv = cv2.cvtColor(self.blurred, cv2.COLOR_BGR2HSV)
        # Select HSV color bounds for mask and create plant mask
        # Hue range: [0,179], Saturation range: [0,255], Value range: [0,255]
        HSV_min = [self.params.parameters['H'][0],
                   self.params.parameters['S'][0],
                   self.params.parameters['V'][0]]
        HSV_max = [self.params.parameters['H'][1],
                   self.params.parameters['S'][1],
                   self.params.parameters['V'][1]]
        if HSV_min[0] > HSV_max[0]:
            hsv_btwn_min = [0, HSV_min[1], HSV_min[2]]
            hsv_btwn_max = [179, HSV_max[1], HSV_max[2]]
            mask_L = cv2.inRange(hsv, np.array(hsv_btwn_min),
                                 np.array(HSV_max))
            mask_U = cv2.inRange(hsv, np.array(HSV_min),
                                 np.array(hsv_btwn_max))
            self.masked = cv2.addWeighted(mask_L, 1.0, mask_U, 1.0, 0.0)
        else:
            self.masked = cv2.inRange(hsv, np.array(HSV_min),
                                      np.array(HSV_max))
        self.image = self.masked.copy()
        self.status['mask'] = True

    def mask2(self):
        """Show regions of original image selected by mask"""
        self.masked2 = cv2.bitwise_and(self.output,
                                       self.output,
                                       mask=self.masked)
        temp = self.image
        self.image = self.masked2
        self.save_annotated('masked2')
        self.image = temp

    def morph(self):
        """Process mask to try to make plants more coherent"""
        if self.params.array is None:
            # Single morphological transformation
            kernel_type = self.params.kt[self.params.kernel_type]
            kernel = cv2.getStructuringElement(kernel_type,
                                               (self.params.parameters['morph'],
                                                self.params.parameters['morph']))
            morph_type = self.params.mt[self.params.morph_type]
            self.morphed = cv2.morphologyEx(self.masked,
                                            morph_type, kernel,
                                            iterations=self.params.parameters[
                                                'iterations'])
        else:
            # List of morphological transformations
            processes = self.params.array
            self.morphed = self.masked
            for process in processes:
                morph_amount = process['size']
                kernel_type = self.params.kt[process['kernel']]
                morph_type = self.params.mt[process['type']]
                iterations = process['iters']
                kernel = cv2.getStructuringElement(kernel_type,
                                                   (morph_amount, morph_amount))
                if morph_type == 'erode':
                    self.morphed = cv2.erode(self.morphed, kernel,
                                             iterations=iterations)
                elif morph_type == 'dilate':
                    self.morphed = cv2.dilate(self.morphed, kernel,
                                              iterations=iterations)
                else:
                    self.morphed = cv2.morphologyEx(self.morphed,
                                                    morph_type, kernel,
                                                    iterations=iterations)
        self.image = self.morphed
        self.status['morph'] = True

    def morph2(self):
        """Show regions of original image selected by morph"""
        self.morphed2 = cv2.bitwise_and(self.output,
                                        self.output,
                                        mask=self.morphed)
        temp = self.image
        self.image = self.morphed2
        self.save_annotated('morphed2')
        self.image = temp

    def initial_processing(self):
        """Process image in preparation for detecting plants"""
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
            self.mask2()
        if self.calibration_debug:
            self.show()

        # Transform mask to try to make objects more coherent
        self.morph()
        if self.debug:
            self.save_annotated('morphed')
            self.morph2()
        if self.calibration_debug:
            self.show()

    def clump_buster(self):
        """Break up selected regions of morphed image into smaller regions
           by splitting them into quarters"""
        clump_img = self.morphed.copy()
        try:
            contours, hierarchy = cv2.findContours(clump_img,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _img, contours, hierarchy = cv2.findContours(clump_img,
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cnt = contours[i]
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            cv2.line(self.morphed, (rx + rw / 2, ry), (rx + rw / 2, ry + rh),
                     (0), rw / 7)
            cv2.line(self.morphed, (rx, ry + rh / 2), (rx + rw, ry + rh / 2),
                     (0), rh / 7)
        kernel = cv2.getStructuringElement(self.params.kt['ellipse'],
                                           (self.params.parameters['morph'],
                                            self.params.parameters['morph']))
        self.morphed = cv2.dilate(self.morphed, kernel, iterations=1)
        self.image = self.morphed
        self.status['bust'] = True

    def grey(self):
        """Grey out region in output image not selected by morphed mask"""
        grey_bg = cv2.addWeighted(np.full_like(self.output, 255),
                                  0.4, self.output, 0.6, 0)
        black_fg = cv2.bitwise_and(grey_bg, grey_bg,
                                   mask=cv2.bitwise_not(self.morphed))
        plant_fg_grey_bg = cv2.add(cv2.bitwise_and(self.output, self.output,
                                                   mask=self.morphed), black_fg)
        self.greyed = plant_fg_grey_bg.copy()
        self.output = self.greyed
        self.marked = self.greyed
        self.status['grey'] = True

    def find(self, **kwargs):
        """Create contours, find locations of objects, and mark them.
           Requires morphed image."""
        small_c = False  # default
        circle = True  # default
        draw_contours = True  # default
        calibration = False  # default
        for key in kwargs:
            if key == 'small_c':
                small_c = kwargs[key]
            if key == 'circle':
                circle = kwargs[key]
            if key == 'draw_contours':
                draw_contours = kwargs[key]
            if key == 'calibration':
                calibration = kwargs[key]
        # Find contours (hopefully of outside edges of plants)
        self.contoured = self.morphed.copy()
        try:
            contours, hierarchy = cv2.findContours(self.contoured,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _img, contours, hierarchy = cv2.findContours(self.contoured,
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)
            self.contoured = np.zeros_like(self.contoured, np.uint8)
        self.db.object_count = len(contours)
        if calibration and self.db.object_count > 10 and 0:
            print("ERROR: Too many calibration objects detected in image.")
            print("\t ({}) Try changing calibration parameters.".format(
                self.db.object_count))
            self.params.print_()
            cv2.drawContours(self.output, contours, -1, (255, 0, 0), 3)
            self.image = self.output
            self.show()
            sys.exit(0)

        # Loop through contours
        self.db.pixel_locations = []
        only_one_object = False
        for i, cnt in enumerate(contours):
            # Calculate plant location by using centroid of contour
            M = cv2.moments(cnt)
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                (mcx, mcy), radius = cv2.minEnclosingCircle(cnt)
            except ZeroDivisionError:
                continue

            if small_c:
                radius = 20
            if calibration:
                # Mark calibration object with blue circle
                center = (int(mcx), int(mcy))
                cv2.circle(self.marked, center, int(radius), (255, 0, 0), 4)

            # Draw contours
            if calibration:
                cv2.drawContours(self.marked, [cnt], 0, (0, 255, 0), 3)
            else:
                cv2.drawContours(self.contoured, [cnt], 0, (255, 255, 255), 3)
                cv2.drawContours(self.marked, [cnt], 0, (0, 0, 0), 6)
                cv2.drawContours(self.marked, [cnt], 0, (255, 255, 255), 2)

            self.db.pixel_locations.append([cx, cy, radius])
            if calibration:
                if i == 0:
                    self.db.calibration_pixel_locations = [mcx, mcy, radius]
                    only_one_object = True
                else:
                    try:
                        self.db.calibration_pixel_locations = np.vstack(
                            (self.db.calibration_pixel_locations,
                             [mcx, mcy, radius]))
                        only_one_object = False
                    except ValueError:
                        self.db.calibration_pixel_locations = [
                            mcx, mcy, radius]
                        only_one_object = True
        if calibration and only_one_object:
            self.db.calibration_pixel_locations = [
                self.db.calibration_pixel_locations]
            self.db.object_count = 1
        self.image = self.contoured
        self.status['mark'] = True

    def coordinates(self, p2c):
        """Rotate image according to calibration data, detect objects and
           their coordinates"""
        self.rotate_main_images(
            p2c.total_rotation_angle)  # rotate according to calibration
        self.image = self.morphed
        self.find()  # detect pixel locations of objects
        p2c.p2c(self.db)  # convert pixel locations to coordinates

    def label(self, p2c=None):
        """Draw circles on image indicating detected plants"""
        def circle(color):
            c = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
            for obj in self.db.pixel_locations:
                cv2.circle(self.marked, (int(obj[0]), int(obj[1])),
                           int(obj[2]), c[color], 4)

        if p2c is None:
            circle('red')
        else:
            known = [[_['x'], _['y'], _['radius']] for _
                     in self.db.plants['known']]
            self.db.coordinate_locations = known
            p2c.c2p(self.db)
            circle('green')
            remove = [[_['x'], _['y'], _['radius']] for _
                      in self.db.plants['remove']]
            self.db.coordinate_locations = remove
            p2c.c2p(self.db)
            circle('red')
            save = [[_['x'], _['y'], _['radius']] for _
                    in self.db.plants['save']]
            self.db.coordinate_locations = save
            p2c.c2p(self.db)
            circle('blue')

    def grid(self, p2c):
        """Draw grid on image indicating coordinate system"""
        w = self.marked.shape[1]
        textsize = w / 2000.
        textweight = int(3.5 * textsize)

        def grid_point(point, pointtype):
            if pointtype == 'coordinates':
                if len(point) < 3:
                    point = list(point) + [0]
                self.db.coordinate_locations = point
                p2c.c2p(self.db)
                point_pixel_location = np.array(self.db.pixel_locations)[0]
                x = point_pixel_location[0]
                y = point_pixel_location[1]
            else:  # pixels
                x = point[0]
                y = point[1]
            tps = w / 300.
            # crosshair center
            self.marked[int(y - tps):int(y + tps + 1),
                        int(x - tps):int(x + tps + 1)] = (255, 255, 255)
            # crosshair lines
            self.marked[int(y - tps * 4):int(y + tps * 4 + 1),
                        int(x - tps / 4):int(x + tps / 4 + 1)] = (255,
                                                                  255,
                                                                  255)
            self.marked[int(y - tps / 4):int(y + tps / 4 + 1),
                        int(x - tps * 4):int(x + tps * 4 + 1)] = (255,
                                                                  255,
                                                                  255)
        # grid_point([1650, 2050, 0], 'coordinates')  # test point
        grid_point(p2c.test_coordinates, 'coordinates')  # UTM location
        grid_point(p2c.calibration_params['center_pixel_location'],
                   'pixels')  # image center

        grid_range = np.array([[x] for x in range(0, 20000, 100)])
        large_grid = np.hstack((grid_range, grid_range, grid_range))
        self.db.coordinate_locations = large_grid
        p2c.c2p(self.db)
        large_grid_pl = np.array(self.db.pixel_locations)
        for x, xc in zip(large_grid_pl[:, 0], large_grid[:, 0]):
            if x > self.marked.shape[1] or x < 0:
                continue
            self.marked[:, int(x):int(x + 1)] = (255, 255, 255)
            cv2.putText(self.marked, str(xc), (int(x), 100),
                        cv2.FONT_HERSHEY_SIMPLEX, textsize,
                        (255, 255, 255), textweight)
        for y, yc in zip(large_grid_pl[:, 1], large_grid[:, 1]):
            if y > self.marked.shape[0] or y < 0:
                continue
            self.marked[int(y):int(y + 1), :] = (255, 255, 255)
            cv2.putText(self.marked, str(yc), (100, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, textsize,
                        (255, 255, 255), textweight)
        self.image = self.marked

    def annotate(self):
        """Annotate image with processing parameters"""
        tc = {'white': (255, 255, 255), 'black': (0, 0, 0)}
        bc = {'white': 255, 'black': 0}
        text_color = tc['white']
        bg_color = bc['black']
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.status['blur']:
            lines = ["blur kernel size = {}".format(
                self.params.parameters['blur'])]
        else:
            return self.image
        if self.status['mask']:
            HSV_min = [self.params.parameters['H'][0],
                       self.params.parameters['S'][0],
                       self.params.parameters['V'][0]]
            HSV_max = [self.params.parameters['H'][1],
                       self.params.parameters['S'][1],
                       self.params.parameters['V'][1]]
            lines = lines + [
                "HSV lower bound = {}".format(HSV_min),
                "HSV upper bound = {}".format(HSV_max)]
            if self.status['morph'] and not self.params.array:  # single morph
                lines = lines + [
                    "kernel type = {}".format(self.params.kernel_type),
                    "kernel size = {}".format(self.params.parameters['morph']),
                    "morphological transformation = {}".format(
                        self.params.morph_type),
                    "number of iterations = {}".format(
                        self.params.parameters['iterations'])]
        h = self.image.shape[0]
        w = self.image.shape[1]
        textsize = w / 1200.
        lineheight = int(40 * textsize)
        textweight = int(3.5 * textsize)
        add = lineheight + lineheight * len(lines)
        if self.status['morph'] and self.params.array:  # multiple morphs
            add_1 = add
            add += lineheight + lineheight * len(self.params.array)
        try:  # color image?
            c = self.image.shape[2]
            new_shape = (h + add, w, c)
        except IndexError:
            new_shape = (h + add, w)
        annotated_image = np.full(new_shape, bg_color, np.uint8)
        annotated_image[add:, :] = self.image
        for o, line in enumerate(lines):
            cv2.putText(annotated_image, line,
                        (10, lineheight + o * lineheight),
                        font, textsize, text_color, textweight)
        if self.status['morph'] and self.params.array:  # multiple morphs
            for o, line in enumerate(self.params.array):
                cv2.putText(annotated_image, str(line),
                            (10, add_1 + o * lineheight),
                            font, textsize, text_color, textweight)
        self.status['annotate'] = True
        return annotated_image

if __name__ == "__main__":
    image = Image(Parameters(), DB())

    image.capture()
    image.show()

    if len(sys.argv) == 1:
        directory = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        soil_image = directory + 'soil_image.jpg'
    else:
        soil_image = sys.argv[1]
    image.load(soil_image)
    image.save('loaded')

    image.blur()
    image.mask()
    image.morph()
    image.save('processed')
