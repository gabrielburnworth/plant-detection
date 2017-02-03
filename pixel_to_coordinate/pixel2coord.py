"""Image Pixel Location to Machine Coordinate Conversion."""

from time import sleep
import cv2
import numpy as np
import os


class Pixel2coord():
    """Calibrates the conversion of pixel locations to machine coordinates
    in images. Finds object coordinates in image.
    """
    def __init__(self, calibration_image=None):
        self.image = None
        if calibration_image is not None:
            if isinstance(calibration_image, str):
                self.readimage(calibration_image)
            else:
                self.image = calibration_image
        self.camera_rotation = 0
        if self.camera_rotation > 0:
            self.image = np.rot90(self.image)
        self.proc = None
        self.circled = None
        self.calibration_object_pixel_locations = []
        self.rotationangle = 0
        self.test_rotation = 5  # for testing, add some image rotation
        self.viewoutputimage = False  # overridden as True if running script
        self.coord_scale = None
        self.total_rotation_angle = 0
        self.center_pixel_location = None
        # Parameters imported from file (or defaults below)
        self.calibration_circles_xaxis = None
        self.image_bot_origin_location = None
        self.calibration_circle_separation = None
        self.camera_offset_coordinates = None
        self.iterations = None
        self.test_coordinates = None
        self.blur_amount = None
        self.morph_amount = None
        self.HSV_min = None
        self.HSV_max = None
        self.output_text = True
        self.tmp_dir = None

        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.parameters_filename = "/pixel2coord_calibration_parameters.txt"
        self.parameters_filepath = self.dir + self.parameters_filename
        self.load_calibration_parameters()

        # Run calibration sequence for provided image
        if self.image is not None:
            self.calibration()

    def getcoordinates(self):
        """Get machine coordinates from bot."""
        # For now, return testing coordintes:
        return self.test_coordinates

    def save_calibration_parameters(self):
        """Save calibration parameters to file."""
        if self.tmp_dir is None:
            filename = self.parameters_filepath
        else:
            filename = self.tmp_dir + self.parameters_filename

        with open(filename, 'w') as f:
            f.write('calibration_circles_xaxis {}\n'.format(
                [1 if self.calibration_circles_xaxis else 0][0]))
            f.write('image_bot_origin_location {} {}\n'.format(
                *self.image_bot_origin_location))
            f.write('calibration_circle_separation {}\n'.format(
                self.calibration_circle_separation))
            f.write('camera_offset_coordinates {} {}\n'.format(
                *self.camera_offset_coordinates))
            f.write('iterations {}\n'.format(
                self.iterations))
            f.write('test_coordinates {} {}\n'.format(
                *self.test_coordinates))
            f.write('blur_amount {}\n'.format(
                self.blur_amount))
            f.write('morph_amount {}\n'.format(
                self.morph_amount))
            f.write('HSV_min {} {} {}\n'.format(
                *self.HSV_min))
            f.write('HSV_max {} {} {}\n'.format(
                *self.HSV_max))
            f.write('coord_scale {}\n'.format(
                self.coord_scale))
            f.write('total_rotation_angle {}\n'.format(
                self.total_rotation_angle))
            f.write('center_pixel_location {} {}\n'.format(
                *self.center_pixel_location))

    def load_calibration_parameters(self):
        """Load calibration parameters from file
        or use defaults and save to file."""
        if self.tmp_dir is None:
            filename = self.parameters_filepath
        else:
            filename = self.tmp_dir + self.parameters_filename
        try:  # Load calibration parameters from file
            with open(filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                if "calibration_circles_xaxis" in line:
                    self.calibration_circles_xaxis = int(line[1])
                if "image_bot_origin_location" in line:
                    self.image_bot_origin_location = [int(line[1]),
                                                      int(line[2])]
                if "calibration_circle_separation" in line:
                    self.calibration_circle_separation = float(line[1])
                if "camera_offset_coordinates" in line:
                    self.camera_offset_coordinates = [float(line[1]),
                                                      float(line[2])]
                if "iterations" in line:
                    self.iterations = int(line[1])
                if "test_coordinates" in line:
                    self.test_coordinates = [float(line[1]),
                                             float(line[2])]
                if "blur_amount" in line:
                    self.blur_amount = int(line[1])
                    if self.blur_amount % 2 == 0:
                        self.blur_amount += 1
                if "morph_amount" in line:
                    self.morph_amount = int(line[1])
                if "HSV_min" in line:
                    self.HSV_min = [float(line[1]),
                                    float(line[2]),
                                    float(line[3])]
                if "HSV_max" in line:
                    self.HSV_max = [float(line[1]),
                                    float(line[2]),
                                    float(line[3])]
                if "coord_scale" in line:
                    self.coord_scale = float(line[1])
                if "total_rotation_angle" in line:
                    self.total_rotation_angle = float(line[1])
                if "center_pixel_location" in line:
                    self.center_pixel_location = [float(line[1]),
                                                  float(line[2])]
        except IOError:  # Use defaults and save to file
            if self.image is not None:
                self.calibration_circles_xaxis = True  # calib. circles along xaxis
                self.image_bot_origin_location = [0, 1]  # image bot axes locations
                self.calibration_circle_separation = 1000  # distance btwn red dots
                self.camera_offset_coordinates = [200, 100]  # UTM camera offset
                self.iterations = 3  # min 2 if image rotated or if rotation unkwn
                self.test_coordinates = [600, 400]  # calib image coord. location
                self.blur_amount = 5  # must be odd
                self.morph_amount = 15
                self.HSV_min = [160, 100, 100]  # to wrap (reds), use H_min > H_max
                self.HSV_max = [20, 255, 255]
                self.center_pixel_location = np.array(self.image.shape[:2][::-1]) / 2

                try:
                    self.save_calibration_parameters()
                except IOError:
                    self.tmp_dir = "/tmp"
                    self.save_calibration_parameters()
            else:
                self.tmp_dir = "/tmp"
                self.load_calibration_parameters()

    def readimage(self, filename):
        """Read an image from a file."""
        self.image = cv2.imread(filename)

    def showimage(self, image_to_show):
        """Show an image."""
        cv2.imshow("image", image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rotationdetermination(self):
        """Determine angle of rotation if necessary."""
        threshold = 0
        obj_1_x, obj_1_y, _ = self.calibration_object_pixel_locations[0]
        obj_2_x, obj_2_y, _ = self.calibration_object_pixel_locations[1]
        if not self.calibration_circles_xaxis:
            if obj_1_x > obj_2_x:
                obj_1_x, obj_2_x = obj_2_x, obj_1_x
                obj_1_y, obj_2_y = obj_2_y, obj_1_y
        dx = (obj_1_x - obj_2_x)
        dy = (obj_1_y - obj_2_y)
        if self.calibration_circles_xaxis:
            difference = abs(dy)
            trig = difference / dx
        else:
            difference = abs(dx)
            trig = difference / dy
        if difference > threshold:
            rotation_angle_radians = np.tan(trig)
            self.rotationangle = 180. / np.pi * rotation_angle_radians

    def rotateimage(self, rotationangle):
        """Rotate image number of degrees."""
        try:
            rows, cols, _ = self.image.shape
        except ValueError:
            rows, cols = self.image.shape
        mtrx = cv2.getRotationMatrix2D((int(cols / 2), int(rows / 2)),
                                       rotationangle, 1)
        self.image = cv2.warpAffine(self.image, mtrx, (cols, rows))

    def process(self):
        """Prepare image for contour detection."""
        blur = cv2.medianBlur(self.image, self.blur_amount)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
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
        self.proc = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                     np.ones((self.morph_amount,
                                              self.morph_amount), np.uint8))

    def findobjects(self, **kwargs):
        """Create contours and find locations of objects."""
        small_c = False  # default
        circle = True  # default
        draw_contours = True  # default
        for key in kwargs:
            if key == 'small_c': small_c = kwargs[key]
            if key == 'circle': circle = kwargs[key]
            if key == 'draw_contours': draw_contours = kwargs[key]
        try:
            contours, _ = cv2.findContours(
                          self.proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours, _ = cv2.findContours(
                          self.proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.circled = self.image.copy()
        for i, cnt in enumerate(contours):
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if i == 0:
                self.calibration_object_pixel_locations = [cx, cy, radius]
            else:
                self.calibration_object_pixel_locations = np.vstack(
                    (self.calibration_object_pixel_locations,
                     [cx, cy, radius]))
            center = (int(cx), int(cy))
            if small_c:
                radius = 20
            if circle:
                cv2.circle(self.circled, center, int(radius), (255, 0, 0), 4)
            if draw_contours:
                cv2.drawContours(self.proc, [cnt], 0, (255, 255, 255), 3)
                cv2.drawContours(self.circled, [cnt], 0, (0, 255, 0), 3)

    def calibrate(self):
        """Determine coordinate conversion parameters."""
        if len(self.calibration_object_pixel_locations) > 1:
            calibration_circle_sep = float(self.calibration_circle_separation)
            if self.calibration_circles_xaxis:
                i = 0
            else:
                i = 1
            object_sep = abs(self.calibration_object_pixel_locations[0][i] -
                             self.calibration_object_pixel_locations[1][i])
            self.coord_scale = calibration_circle_sep / object_sep

    def p2c(self, object_pixel_locations):
        """Convert pixel locations to machine coordinates from image center."""
        object_pixel_locations = np.array(object_pixel_locations)
        if len(object_pixel_locations) == 0:
            return [], []
        try:
            object_pixel_locations.shape[1]
        except IndexError:
            object_pixel_locations = [object_pixel_locations]
        object_pixel_locations = np.array(object_pixel_locations)
        coord = np.array(self.getcoordinates(), dtype=float)
        camera_offset = np.array(self.camera_offset_coordinates, dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        sign = [1 if s == 1 else -1 for s in self.image_bot_origin_location]
        coord_scale = np.array([self.coord_scale, self.coord_scale])
        object_coordinates = []
        if self.output_text:
            print("Detected object machine coordinates ( X Y ) with R = radius:")
        for o, object_pixel_location in enumerate(object_pixel_locations[:, :2]):
            radius = object_pixel_locations[:][o][2]
            moc = (camera_coordinates +
                   sign * coord_scale *
                   (self.center_pixel_location - object_pixel_location))
            if self.output_text:
                print("    ( {:5.0f} {:5.0f} ) R = {R:.0f}".format(*moc, R=radius))
            object_coordinates.append(
                [moc[0], moc[1], coord_scale[0] * radius])
        return object_coordinates, object_pixel_locations

    def c2p(self, object_coordinates):
        """Convert machine coordinates to pixel locations
        using image center."""
        object_coordinates = np.array(object_coordinates)
        if len(object_coordinates) == 0:
            return []
        try:
            object_coordinates.shape[1]
        except IndexError:
            object_coordinates = [object_coordinates]
        object_coordinates = np.array(object_coordinates)
        coord = np.array(self.getcoordinates(), dtype=float)
        camera_offset = np.array(self.camera_offset_coordinates, dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        center_pixel_location = self.center_pixel_location[:2]
        sign = [1 if s == 1 else -1 for s in self.image_bot_origin_location]
        coord_scale = np.array([self.coord_scale, self.coord_scale])
        object_pixel_locations = []
        for o, object_coordinate in enumerate(object_coordinates[:, :2]):
            opl = (center_pixel_location -
                   ((object_coordinate - camera_coordinates)
                    / (sign * coord_scale)))
            object_pixel_locations.append([opl[0], opl[1],
                                           object_coordinates[o][2]
                                           / coord_scale[0]])
        return object_pixel_locations

    def calibration(self):
        """Determine pixel to coordinate conversion scale
        and image rotation angle."""
        self.total_rotation_angle = 0
        for i in range(0, self.iterations):
            self.process()
            self.findobjects()
            if i != (self.iterations - 1):
                self.rotationdetermination()
                self.rotateimage(self.rotationangle)
                self.total_rotation_angle += self.rotationangle
        if self.total_rotation_angle != 0:
            print(" Note: required rotation executed = {:.2f} degrees".format(
                self.total_rotation_angle))
        self.calibrate()
        if self.viewoutputimage:
            self.showimage(self.circled)
        self.save_calibration_parameters()

    def determine_coordinates(self):
        """Use calibration parameters to determine locations of objects."""
        self.rotateimage(self.total_rotation_angle)
        self.process()
        self.findobjects()
        _, _ = self.p2c(self.calibration_object_pixel_locations)
        if self.viewoutputimage:
            self.showimage(self.circled)

if __name__ == "__main__":
    folder = os.path.dirname(os.path.realpath(__file__))
    P2C = Pixel2coord(calibration_image=folder + "/p2c_test_calibration.jpg")
    P2C.viewoutputimage = True
    # Calibration
    P2C.rotateimage(P2C.test_rotation)
    P2C.calibration()
    # Tests
    # Object detection
    P2C.readimage(folder + "/p2c_test_objects.jpg")
    P2C.rotateimage(P2C.test_rotation)
    P2C.determine_coordinates()
    # Color range
    P2C.readimage(folder + "/p2c_test_color.jpg")
    P2C.process()
    P2C.findobjects(circle=False)
    if P2C.viewoutputimage:
        P2C.showimage(P2C.circled)
