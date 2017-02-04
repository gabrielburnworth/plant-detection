"""Image Pixel Location to Machine Coordinate Conversion."""

from time import sleep
import cv2
import numpy as np
import os
from Parameters import Parameters
from Capture import Capture
from Image import Image
from DB import DB

class Pixel2coord():
    """Calibrates the conversion of pixel locations to machine coordinates
    in images. Finds object coordinates in image.
    """
    def __init__(self, db, calibration_image=None):
        self.cparams = Parameters()
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.parameters_file = "pixel2coord_calibration_parameters.txt"
        if calibration_image is not None:
            self.temp_cimage = Image(Parameters(), DB())
            self.temp_cimage.load(calibration_image)

        # Parameters imported from file (or defaults below)
        self.coord_scale = None
        self.total_rotation_angle = 0
        self.center_pixel_location = None
        self.calibration_circles_xaxis = None
        self.image_bot_origin_location = None
        self.calibration_circle_separation = None
        self.camera_offset_coordinates = None
        self.iterations = None
        self.test_coordinates = None

        self.db = db
        self.load_calibration_parameters()

        if calibration_image is not None:
            self.image = Image(self.cparams, self.db)
            self.image.load(calibration_image)
        self.camera_rotation = 0
        if self.camera_rotation > 0:
            self.image.image = np.rot90(self.image)
        self.proc = None
        self.circled = None
        self.rotationangle = 0
        self.test_rotation = 5  # for testing, add some image rotation
        self.viewoutputimage = False  # overridden as True if running script
        self.output_text = True
        self.get_bot_coordinates = Capture()._getcoordinates

        # Run calibration sequence for provided image
        if calibration_image is not None:
            self.calibration()

    def save_calibration_parameters(self):
        """Save calibration parameters to file."""
        if self.db.tmp_dir is None:
            directory = self.dir
        else:
            directory = self.db.tmp_dir
        self.cparams.save(directory, self.parameters_file)
        with open(directory + self.parameters_file, 'a') as f:
            f.write('calibration_circles_xaxis {}\n'.format(
                [1 if self.calibration_circles_xaxis else 0][0]))
            f.write('image_bot_origin_location {} {}\n'.format(
                *self.image_bot_origin_location))
            f.write('calibration_circle_separation {}\n'.format(
                self.calibration_circle_separation))
            f.write('camera_offset_coordinates {} {}\n'.format(
                *self.camera_offset_coordinates))
            f.write('rotation_iters {}\n'.format(
                self.iterations))
            f.write('test_coordinates {} {}\n'.format(
                *self.test_coordinates))
            f.write('coord_scale {}\n'.format(
                self.coord_scale))
            f.write('total_rotation_angle {}\n'.format(
                self.total_rotation_angle))
            f.write('center_pixel_location {} {}\n'.format(
                *self.center_pixel_location))

    def load_calibration_parameters(self):
        """Load calibration parameters from file
        or use defaults and save to file."""
        def load(directory):  # Load calibration parameters from file
            self.cparams.load(directory, self.parameters_file)
            with open(directory + self.parameters_file, 'r') as f:
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
                    if "rotation_iters" in line:
                        self.iterations = int(line[1])
                    if "test_coordinates" in line:
                        self.test_coordinates = [float(line[1]),
                                                 float(line[2])]
                    if "coord_scale" in line:
                        self.coord_scale = float(line[1])
                    if "total_rotation_angle" in line:
                        self.total_rotation_angle = float(line[1])
                    if "center_pixel_location" in line:
                        self.center_pixel_location = [float(line[1]),
                                                      float(line[2])]
                essential_parameters = [self.camera_offset_coordinates,
                                        self.image_bot_origin_location,
                                        self.coord_scale,
                                        self.center_pixel_location]
                if any(param is None for param in essential_parameters):
                    raise IOError
        try:
            try:
                load(self.dir)
            except IOError:
                self.db.tmp_dir = "/tmp/"
                load(self.db.tmp_dir)
        except IOError:  # Use defaults and save to file
            self.calibration_circles_xaxis = True  # calib. circles along xaxis
            self.image_bot_origin_location = [0, 1]  # image bot axes locations
            self.calibration_circle_separation = 1000  # distance btwn red dots
            self.camera_offset_coordinates = [200, 100]  # UTM camera offset
            self.iterations = 3  # min 2 if image rotated or if rotation unkwn
            self.test_coordinates = [600, 400]  # calib image coord. location
            self.cparams.blur_amount = 5  # must be odd
            self.cparams.morph_amount = 15
            self.cparams.HSV_min = [160, 100, 100]  # to wrap (reds), use H_min > H_max
            self.cparams.HSV_max = [20, 255, 255]
            self.center_pixel_location = np.array(self.temp_cimage.image.shape[:2][::-1]) / 2

    def rotationdetermination(self):
        """Determine angle of rotation if necessary."""
        threshold = 0
        obj_1_x, obj_1_y, _ = self.db.calibration_pixel_locations[0]
        obj_2_x, obj_2_y, _ = self.db.calibration_pixel_locations[1]
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

    def process(self):
        """Prepare image for contour detection."""
        self.image.blur()
        self.image.mask()
        self.image.morph()

    def calibrate(self):
        """Determine coordinate conversion parameters."""
        if len(self.db.calibration_pixel_locations) > 1:
            calibration_circle_sep = float(self.calibration_circle_separation)
            if self.calibration_circles_xaxis:
                i = 0
            else:
                i = 1
            object_sep = abs(self.db.calibration_pixel_locations[0][i] -
                             self.db.calibration_pixel_locations[1][i])
            self.coord_scale = calibration_circle_sep / object_sep

    def p2c(self, db):
        """Convert pixel locations to machine coordinates from image center."""
        db.pixel_locations = np.array(db.pixel_locations)
        if len(db.pixel_locations) == 0:
            return [], []
        try:
            db.pixel_locations.shape[1]
        except IndexError:
            db.pixel_locations = [db.pixel_locations]
        db.pixel_locations = np.array(db.pixel_locations)
        coord = np.array(self.get_bot_coordinates(), dtype=float)
        camera_offset = np.array(self.camera_offset_coordinates, dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        sign = [1 if s == 1 else -1 for s in self.image_bot_origin_location]
        coord_scale = np.array([self.coord_scale, self.coord_scale])
        db.coordinate_locations = []
        for o, object_pixel_location in enumerate(db.pixel_locations[:, :2]):
            radius = db.pixel_locations[:][o][2]
            moc = (camera_coordinates +
                   sign * coord_scale *
                   (self.center_pixel_location - object_pixel_location))
            db.coordinate_locations.append(
                [moc[0], moc[1], coord_scale[0] * radius])

    def c2p(self, db):
        """Convert machine coordinates to pixel locations
        using image center."""
        db.coordinate_locations = np.array(db.coordinate_locations)
        if len(db.coordinate_locations) == 0:
            db.pixel_locations = []
            return
        try:
            db.coordinate_locations.shape[1]
        except IndexError:
            db.coordinate_locations = [db.coordinate_locations]
        db.coordinate_locations = np.array(db.coordinate_locations)
        coord = np.array(self.get_bot_coordinates(), dtype=float)
        camera_offset = np.array(self.camera_offset_coordinates, dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        center_pixel_location = self.center_pixel_location[:2]
        sign = [1 if s == 1 else -1 for s in self.image_bot_origin_location]
        coord_scale = np.array([self.coord_scale, self.coord_scale])
        db.pixel_locations = []
        for o, object_coordinate in enumerate(db.coordinate_locations[:, :2]):
            opl = (center_pixel_location -
                   ((object_coordinate - camera_coordinates)
                    / (sign * coord_scale)))
            db.pixel_locations.append([opl[0], opl[1],
                                           db.coordinate_locations[o][2]
                                           / coord_scale[0]])

    def calibration(self):
        """Determine pixel to coordinate conversion scale
        and image rotation angle."""
        self.total_rotation_angle = 0
        for i in range(0, self.iterations):
            self.process()
            self.image.find(calibration=True)
            if i != (self.iterations - 1):
                self.rotationdetermination()
                self.image.rotate(self.rotationangle)
                self.total_rotation_angle += self.rotationangle
        if self.total_rotation_angle != 0:
            print(" Note: required rotation executed = {:.2f} degrees".format(
                self.total_rotation_angle))
        self.db.print_count(calibration=True)  # print number of objects detected
        self.calibrate()
        self.db.print_coordinates()
        if self.viewoutputimage:
            self.image.image = self.image.marked
            self.image.show()
        self.save_calibration_parameters()

    def determine_coordinates(self):
        """Use calibration parameters to determine locations of objects."""
        self.image.rotate(self.total_rotation_angle)
        self.process()
        self.image.find(calibration=True)
        self.db.print_count(calibration=True)  # print number of objects detected
        self.p2c(self.db)
        self.db.print_coordinates()
        if self.viewoutputimage:
            self.image.image = self.image.marked
            self.image.show()

if __name__ == "__main__":
    folder = os.path.dirname(os.path.realpath(__file__)) + os.sep
    print("Calibration image load...")
    P2C = Pixel2coord(DB(), calibration_image=folder + "p2c_test_calibration.jpg")
    P2C.viewoutputimage = True
    # Calibration
    P2C.image.rotate(P2C.test_rotation)
    P2C.calibration()
    # Tests
    # Object detection
    print("Calibration object test...")
    P2C.image.load(folder + "p2c_test_objects.jpg")
    P2C.image.rotate(P2C.test_rotation)
    P2C.determine_coordinates()
    # Color range
    print("Calibration color range...")
    P2C.image.load(folder + "p2c_test_color.jpg")
    P2C.process()
    P2C.image.find(circle=False)
    if P2C.viewoutputimage:
        P2C.image.image = P2C.image.marked
        P2C.image.show()
