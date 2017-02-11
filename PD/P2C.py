"""Image Pixel Location to Machine Coordinate Conversion."""

import numpy as np
import sys, os
try:
    from .Parameters import Parameters
    from .Capture import Capture
    from .Image import Image
    from .DB import DB
except:
    from Parameters import Parameters
    from Capture import Capture
    from Image import Image
    from DB import DB

class Pixel2coord():
    """Calibrates the conversion of pixel locations to machine coordinates
    in images. Finds object coordinates in image.
    """
    def __init__(self, db, calibration_image=None):
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.parameters_file = "p2c_calibration_parameters.txt"

        self.coord_scale = None
        self.total_rotation_angle = 0
        self.center_pixel_location = None
        self.calibration_circles_xaxis = None
        self.image_bot_origin_location = None
        self.calibration_circle_separation = None
        self.camera_offset_coordinates = None
        self.iterations = None
        self.test_coordinates = None
        self.params_loaded_from_json = False

        self.db = db
        if self.db.calibration_parameters:
            self.cparams = self.db.calibration_parameters
            self.params_loaded_from_json = True
        else:
            self.cparams = Parameters()
        self.load_calibration_parameters()

        if calibration_image is not None:
            self.image = Image(self.cparams, self.db)
            if isinstance(calibration_image, str):
                self.image.load(calibration_image)
            else:
                self.image.image = calibration_image
                try:
                    testfile = 'test_write.try_to_write'
                    f = open(testfile,"w")
                    f.close()
                    os.remove(testfile)
                except IOError:
                    self.image.dir = '/tmp/images/'
                    self.image.save('capture')
                    self.image.load(self.image.dir + 'capture.jpg')
                    self.image.dir = self.dir[:-3]
                else:
                    self.image.save('capture')
                    self.image.load(self.dir[:-3] + 'capture.jpg')
            self.center_pixel_location = np.array(self.image.image.shape[:2][::-1]) / 2
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
        self.debug = False

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
            if not self.params_loaded_from_json:
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
        except IOError:  # Use defaults
            self.db.tmp_dir = None
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
        else:
            self.rotationangle = 0

    def process_image(self):
        """Prepare image for contour detection."""
        if self.debug: self.cparams.print_()
        self.image.blur()
        if self.debug: self.image.show()
        self.image.mask()
        if self.debug: self.image.show()
        self.image.morph()
        if self.debug: self.image.show()

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
            self.process_image()
            self.image.find(calibration=True)
            if len(self.db.calibration_pixel_locations) == 0:
                print("ERROR: Calibration failed. No objects detected.")
                sys.exit(0)
            if i == 0:
                self.db.print_count(calibration=True)  # print number of objects detected
                if self.db.object_count != 2:
                    print(" Warning: {} objects detected. "
                          "Exactly 2 reccomended.".format(self.db.object_count))
            if i != (self.iterations - 1):
                self.rotationdetermination()
                self.image.rotate_main_images(self.rotationangle)
                self.total_rotation_angle += self.rotationangle
        if self.total_rotation_angle != 0:
            print(" Note: required rotation executed = {:.2f} degrees".format(
                self.total_rotation_angle))
        self.calibrate()
        self.db.print_coordinates()
        if self.viewoutputimage:
            self.image.image = self.image.marked
            self.image.show()
        self.save_calibration_parameters()
        if self.params_loaded_from_json:
            self.image.image = self.image.marked
            self.image.save('calibration_result')

    def determine_coordinates(self):
        """Use calibration parameters to determine locations of objects."""
        self.image.rotate_main_images(self.total_rotation_angle)
        self.process_image()
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
    P2C.image.rotate_main_images(P2C.test_rotation)
    P2C.calibration()
    # Tests
    # Object detection
    print("Calibration object test...")
    P2C.image.load(folder + "p2c_test_objects.jpg")
    P2C.image.rotate_main_images(P2C.test_rotation)
    P2C.determine_coordinates()
    # Color range
    print("Calibration color range...")
    P2C.image.load(folder + "p2c_test_color.jpg")
    P2C.process_image()
    P2C.image.find(circle=False)
    if P2C.viewoutputimage:
        P2C.image.image = P2C.image.marked
        P2C.image.show()
