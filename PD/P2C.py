"""Image Pixel Location to Machine Coordinate Conversion."""

import sys
import os
import json
import numpy as np
try:
    from .Parameters import Parameters
    from .Capture import Capture
    from .Image import Image
    from .DB import DB
except Exception:
    from Parameters import Parameters
    from Capture import Capture
    from Image import Image
    from DB import DB
from PD import CeleryPy


class Pixel2coord(object):
    """Image pixel to machine coordinate conversion.

    Calibrates the conversion of pixel locations to machine coordinates
    in images. Finds object coordinates in image.
    """

    def __init__(self, plant_db,
                 calibration_image=None, calibration_data=None):
        """Set initial attributes.

        Arguments:
            Database() instance

        Optional Keyword Arguments:
            calibration_image: filename (str) or image object (default: None)
            calibration_data: P2C().calibration_params JSON,
                              or 'file' or 'env_var' string
                              (default: None)
        """
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.parameters_file = "plant-detection_calibration_parameters.json"

        self.calibration_params = {}
        self.coord_scale = None
        self.total_rotation_angle = 0
        self.center_pixel_location = None
        self.test_coordinates = None
        self.debug = False
        self.env_var_name = 'PLANT_DETECTION_calibration'
        self.plant_db = plant_db
        self.defaults = {'blur': 5, 'morph': 15,
                         'H': [160, 20], 'S': [100, 255], 'V': [100, 255],
                         'calibration_circles_xaxis': True,
                         'image_bot_origin_location': [0, 1],
                         'calibration_circle_separation': 1000,
                         'camera_offset_coordinates': [200, 100],
                         'calibration_iters': 3}

        # Data and parameter preparation
        self.cparams = None
        self._calibration_data_preparation(calibration_data)
        # Image preparation
        self.image = None
        self.camera_rotation = 0
        self._calibration_image_preparation(calibration_image)

        self.proc = None
        self.circled = None
        self.rotationangle = 0
        self.test_rotation = 5  # for testing, add some image rotation
        self.test_coordinates = [600, 400]  # calib image coord. location
        self.viewoutputimage = False  # overridden as True if running script
        self.output_text = True
        self.get_bot_coordinates = Capture().getcoordinates
        self.json_calibration_data = None

    def _calibration_data_preparation(self, calibration_data):
        if calibration_data is None:  # load defaults
            self.calibration_params = self.defaults
        elif calibration_data == 'file':
            try:
                self.load_calibration_parameters()
            except IOError:
                print("Warning: Calibration data file load failed. "
                      "Using defaults.")
                self.calibration_params = self.defaults
        elif calibration_data == 'env_var':
            try:
                self.calibration_params = json.loads(
                    os.environ[self.env_var_name])
            except (KeyError, ValueError):
                print("Warning: Calibration data env var load failed. "
                      "Using defaults.")
                self.calibration_params = self.defaults
        else:  # load the data provided
            self.calibration_params = calibration_data
            self.initialize_data_keys()

        self.cparams = Parameters()
        self.set_calibration_input_params()

    def _calibration_image_preparation(self, calibration_image):
        if calibration_image is not None:
            self.image = Image(self.cparams, self.plant_db)
            if isinstance(calibration_image, str):
                self.image.load(calibration_image)
            else:
                self.image.image = calibration_image
                try:
                    testfile = 'test_write.try_to_write'
                    tryfile = open(testfile, "w")
                    tryfile.close()
                    os.remove(testfile)
                except IOError:
                    self.image.dir = '/tmp/images/'
                    self.image.save('capture')
                    self.image.load(self.image.dir + 'capture.jpg')
                    self.image.dir = self.dir[:-3]
                else:
                    self.image.save('capture')
                    self.image.load(self.dir[:-3] + 'capture.jpg')
            self.calibration_params['center_pixel_location'] = [
                int(a / 2) for a in self.image.image.shape[:2][::-1]]
            self.image.calibration_debug = self.debug

        if self.camera_rotation > 0:
            self.image.image = np.rot90(self.image)

    def save_calibration_parameters(self):
        """Save calibration parameters to file."""
        if self.plant_db.tmp_dir is None:
            directory = self.dir
        else:
            directory = self.plant_db.tmp_dir
        with open(directory + self.parameters_file, 'w') as oututfile:
            json.dump(self.calibration_params, oututfile)

    def save_calibration_data_to_env(self):
        """Save calibration parameters to environment variable."""
        self.json_calibration_data = CeleryPy.set_user_env(
            self.env_var_name,
            json.dumps(self.calibration_params))
        os.environ[self.env_var_name] = json.dumps(self.calibration_params)

    def initialize_data_keys(self):
        """If using JSON with inputs only, create calibration data keys."""
        def _check_for_key(key):
            try:
                self.calibration_params[key]
            except KeyError:
                self.calibration_params[key] = self.defaults[key]
        calibration_keys = ['calibration_circles_xaxis',
                            'image_bot_origin_location',
                            'calibration_circle_separation',
                            'camera_offset_coordinates',
                            'calibration_iters']
        for key in calibration_keys:
            _check_for_key(key)

    def set_calibration_input_params(self):
        """Set input parameters from calibration parameters."""
        self.cparams.parameters['blur'] = self.calibration_params['blur']
        self.cparams.parameters['morph'] = self.calibration_params['morph']
        self.cparams.parameters['H'] = self.calibration_params['H']
        self.cparams.parameters['S'] = self.calibration_params['S']
        self.cparams.parameters['V'] = self.calibration_params['V']

    def load_calibration_parameters(self):
        """Load calibration parameters from file or use defaults."""
        def _load(directory):  # Load calibration parameters from file
            with open(directory + self.parameters_file, 'r') as inputfile:
                self.calibration_params = json.load(inputfile)
        try:
            _load(self.dir)
        except IOError:
            self.plant_db.tmp_dir = "/tmp/"
            _load(self.plant_db.tmp_dir)

    def rotationdetermination(self):
        """Determine angle of rotation if necessary."""
        threshold = 0
        obj_1_x, obj_1_y, _ = self.plant_db.calibration_pixel_locations[0]
        obj_2_x, obj_2_y, _ = self.plant_db.calibration_pixel_locations[1]
        if not self.calibration_params['calibration_circles_xaxis']:
            if obj_1_x > obj_2_x:
                obj_1_x, obj_2_x = obj_2_x, obj_1_x
                obj_1_y, obj_2_y = obj_2_y, obj_1_y
        obj_dx = (obj_1_x - obj_2_x)
        obj_dy = (obj_1_y - obj_2_y)
        if self.calibration_params['calibration_circles_xaxis']:
            difference = abs(obj_dy)
            trig = difference / obj_dx
        else:
            difference = abs(obj_dx)
            trig = difference / obj_dy
        if difference > threshold:
            rotation_angle_radians = np.tan(trig)
            self.rotationangle = 180. / np.pi * rotation_angle_radians
        else:
            self.rotationangle = 0

    def calibrate(self):
        """Determine coordinate conversion parameters."""
        if len(self.plant_db.calibration_pixel_locations) > 1:
            calibration_circle_sep = float(
                self.calibration_params['calibration_circle_separation'])
            if self.calibration_params['calibration_circles_xaxis']:
                i = 0
            else:
                i = 1
            object_sep = abs(self.plant_db.calibration_pixel_locations[0][i] -
                             self.plant_db.calibration_pixel_locations[1][i])
            self.calibration_params['coord_scale'] = round(
                calibration_circle_sep / object_sep, 4)

    def p2c(self, plant_db):
        """Convert pixel locations to machine coordinates from image center."""
        plant_db.pixel_locations = np.array(plant_db.pixel_locations)
        if len(plant_db.pixel_locations) == 0:
            plant_db.coordinate_locations = []
            return
        try:
            plant_db.pixel_locations.shape[1]
        except IndexError:
            plant_db.pixel_locations = np.vstack(
                [plant_db.pixel_locations])
        coord = np.array(self.get_bot_coordinates()[:2], dtype=float)
        camera_offset = np.array(
            self.calibration_params['camera_offset_coordinates'], dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        sign = [1 if s == 1 else -1 for s
                in self.calibration_params['image_bot_origin_location']]
        coord_scale = np.repeat(self.calibration_params['coord_scale'], 2)
        plant_db.coordinate_locations = []
        for obj_num, object_pixel_location in enumerate(
                plant_db.pixel_locations[:, :2]):
            radius = plant_db.pixel_locations[:][obj_num][2]
            moc = (camera_coordinates +
                   sign * coord_scale *
                   (self.calibration_params['center_pixel_location']
                    - object_pixel_location))
            plant_db.coordinate_locations.append(
                [moc[0], moc[1], coord_scale[0] * radius])

    def c2p(self, plant_db):
        """Convert coordinates to pixel locations using image center."""
        plant_db.coordinate_locations = np.array(plant_db.coordinate_locations)
        if len(plant_db.coordinate_locations) == 0:
            plant_db.pixel_locations = []
            return
        try:
            plant_db.coordinate_locations.shape[1]
        except IndexError:
            plant_db.coordinate_locations = np.vstack(
                [plant_db.coordinate_locations])
        coord = np.array(self.get_bot_coordinates()[:2], dtype=float)
        camera_offset = np.array(
            self.calibration_params['camera_offset_coordinates'], dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        center_pixel_location = self.calibration_params[
            'center_pixel_location'][:2]
        sign = [1 if s == 1 else -1 for s
                in self.calibration_params['image_bot_origin_location']]
        coord_scale = np.repeat(self.calibration_params['coord_scale'], 2)
        plant_db.pixel_locations = []
        for obj_num, object_coordinate in enumerate(
                plant_db.coordinate_locations[:, :2]):
            opl = (center_pixel_location -
                   ((object_coordinate - camera_coordinates)
                    / (sign * coord_scale)))
            plant_db.pixel_locations.append(
                [opl[0], opl[1],
                 plant_db.coordinate_locations[obj_num][2] / coord_scale[0]])

    def calibration(self):
        """Determine pixel to coordinate scale and image rotation angle."""
        total_rotation_angle = 0
        warning_issued = False
        if self.debug:
            self.cparams.print_input()
        for i in range(0, self.calibration_params['calibration_iters']):
            self.image.initial_processing()
            self.image.find(calibration=True)  # find objects
            # If not the last iteration, determine camera rotation angle
            if i != (self.calibration_params['calibration_iters'] - 1):
                # Check number of objects detected and notify user if needed.
                if len(self.plant_db.calibration_pixel_locations) == 0:
                    print("ERROR: Calibration failed. No objects detected.")
                    return True
                if self.plant_db.object_count > 2:
                    if not warning_issued:
                        print(" Warning: {} objects detected. "
                              "Exactly 2 reccomended. "
                              "Incorrect results likely.".format(
                                  self.plant_db.object_count))
                        warning_issued = True
                if self.plant_db.object_count < 2:
                    print(" ERROR: {} objects detected. "
                          "At least 2 required. Exactly 2 reccomended.".format(
                              self.plant_db.object_count))
                    return True
                # Use detected objects to determine required rotation angle
                self.rotationdetermination()
                if abs(self.rotationangle) > 120:
                    print(" ERROR: Excessive rotation required. "
                          "Check that the calibration objects are "
                          "parallel with the desired axis and that "
                          "they are the only two objects detected.")
                    return True
                self.image.rotate_main_images(self.rotationangle)
                total_rotation_angle += self.rotationangle
        self.calibrate()
        fail_flag = self._calibration_output(total_rotation_angle)
        return fail_flag

    def _calibration_output(self, total_rotation_angle):
        if self.viewoutputimage:
            self.image.image = self.image.marked
            self.image.show()
        while abs(total_rotation_angle) > 360:
            if total_rotation_angle < 0:
                total_rotation_angle += 360
            else:
                total_rotation_angle -= 360
        self.calibration_params['total_rotation_angle'] = round(
            total_rotation_angle, 3)
        try:
            self.calibration_params['coord_scale']
            failure_flag = False
        except KeyError:
            print("ERROR: Calibration failed.")
            failure_flag = True
        return failure_flag

    def determine_coordinates(self):
        """Use calibration parameters to determine locations of objects."""
        self.image.rotate_main_images(
            self.calibration_params['total_rotation_angle'])
        if self.debug:
            self.cparams.print_input()
        self.image.initial_processing()
        self.image.find(calibration=True)
        self.plant_db.print_count(calibration=True)  # print detected obj count
        self.p2c(self.plant_db)
        self.plant_db.print_coordinates()
        if self.viewoutputimage:
            self.image.image = self.image.marked
            self.image.show()


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
    print("Calibration image load...")
    P2C = Pixel2coord(DB(), calibration_image=DIR +
                      "p2c_test_calibration.jpg")
    P2C.viewoutputimage = True
    # Calibration
    P2C.image.rotate_main_images(P2C.test_rotation)
    EXIT = P2C.calibration()
    if EXIT:
        sys.exit(0)
    P2C.plant_db.print_count(calibration=True)  # print detected object count
    if P2C.calibration_params['total_rotation_angle'] != 0:
        print(" Note: required rotation executed = {:.2f} degrees".format(
            P2C.calibration_params['total_rotation_angle']))
    # Tests
    # Object detection
    print("Calibration object test...")
    P2C.image.load(DIR + "p2c_test_objects.jpg")
    P2C.image.rotate_main_images(P2C.test_rotation)
    P2C.determine_coordinates()
    # Color range
    print("Calibration color range...")
    P2C.image.load(DIR + "p2c_test_color.jpg")
    P2C.cparams.print_input()
    P2C.image.initial_processing()
    P2C.image.find()
    if P2C.viewoutputimage:
        P2C.image.image = P2C.image.marked
        P2C.image.show()
