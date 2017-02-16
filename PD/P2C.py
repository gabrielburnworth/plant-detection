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
    from .CeleryPy import CeleryPy
except:
    from Parameters import Parameters
    from Capture import Capture
    from Image import Image
    from DB import DB
    from CeleryPy import CeleryPy


class Pixel2coord():
    """Calibrates the conversion of pixel locations to machine coordinates
    in images. Finds object coordinates in image.
    """

    def __init__(self, db, calibration_image=None, env_var=False):
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.parameters_file = "plant-detection_p2c_calibration_parameters.json"

        self.calibration_params = {}
        self.coord_scale = None
        self.total_rotation_angle = 0
        self.center_pixel_location = None
        self.test_coordinates = None
        self.params_loaded_from_json = False

        self.db = db
        if self.db.calibration_parameters:
            self.cparams = self.db.calibration_parameters
            self.params_loaded_from_json = True
        else:
            self.cparams = Parameters()
        if env_var and 0:
            self.load_calibration_parameters_from_env_var()
        else:
            self.load_calibration_parameters()

        if calibration_image is not None:
            self.image = Image(self.cparams, self.db)
            if isinstance(calibration_image, str):
                self.image.load(calibration_image)
            else:
                self.image.image = calibration_image
                try:
                    testfile = 'test_write.try_to_write'
                    f = open(testfile, "w")
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
            self.calibration_params['center_pixel_location'
                                    ] = list(map(int, np.array(
                                        self.image.image.shape[:2][::-1]) / 2))
        self.camera_rotation = 0
        if self.camera_rotation > 0:
            self.image.image = np.rot90(self.image)
        self.proc = None
        self.circled = None
        self.rotationangle = 0
        self.test_rotation = 5  # for testing, add some image rotation
        self.test_coordinates = [600, 400]  # calib image coord. location
        self.viewoutputimage = False  # overridden as True if running script
        self.output_text = True
        self.get_bot_coordinates = Capture().getcoordinates
        self.debug = False

    def save_calibration_parameters(self):
        """Save calibration parameters to file."""
        if self.db.tmp_dir is None:
            directory = self.dir
        else:
            directory = self.db.tmp_dir
        with open(directory + self.parameters_file, 'w') as f:
            json.dump(self.calibration_params, f)

    def save_calibration_data_to_env_var(self):
        """Save calibration parameters to file."""
        CeleryPy().save_calibration_to_env_var(self.calibration_params)

    def set_calibration_parameters_from_defaults(self):
        """Set calibration parameters from defaults."""
        self.calibration_params = {'blur': 5, 'morph': 15,
                                   'H': [160, 20], 'S': [100, 255], 'V': [100, 255],
                                   'calibration_circles_xaxis': True,
                                   'image_bot_origin_location': [0, 1],
                                   'calibration_circle_separation': 1000,
                                   'camera_offset_coordinates': [200, 100],
                                   'calibration_iters': 3}

    def set_input_parameters_for_calibration(self):
        """Set input parameters from calibration parameters."""
        self.cparams.parameters['blur'] = self.calibration_params['blur']
        self.cparams.parameters['morph'] = self.calibration_params['morph']
        self.cparams.parameters['H'] = self.calibration_params['H']
        self.cparams.parameters['S'] = self.calibration_params['S']
        self.cparams.parameters['V'] = self.calibration_params['V']

    def load_calibration_parameters(self):
        """Load calibration parameters from file
        or use defaults."""
        def load(directory):  # Load calibration parameters from file
            with open(directory + self.parameters_file, 'r') as f:
                self.calibration_params = json.load(f)
            essential_parameters = ['camera_offset_coordinates',
                                    'image_bot_origin_location',
                                    'coord_scale',
                                    'total_rotation_angle',
                                    'center_pixel_location']
            for param in essential_parameters:
                try:
                    self.calibration_params[param]
                except KeyError:
                    raise IOError
        try:
            try:
                load(self.dir)
            except IOError:
                self.db.tmp_dir = "/tmp/"
                load(self.db.tmp_dir)
        except IOError:  # Use defaults
            self.db.tmp_dir = None
            self.set_calibration_parameters_from_defaults()
        self.set_input_parameters_for_calibration()

    def load_calibration_parameters_from_env_var(self):
        """Load calibration parameters from environment variable
        or use defaults."""
        try:
            self.calibration_params = json.loads(
                os.environ['PLANT_DETECTION_calibration'])
        except KeyError:
            self.set_calibration_parameters_from_defaults()
        self.set_input_parameters_for_calibration()

    def rotationdetermination(self):
        """Determine angle of rotation if necessary."""
        threshold = 0
        obj_1_x, obj_1_y, _ = self.db.calibration_pixel_locations[0]
        obj_2_x, obj_2_y, _ = self.db.calibration_pixel_locations[1]
        if not self.calibration_params['calibration_circles_xaxis']:
            if obj_1_x > obj_2_x:
                obj_1_x, obj_2_x = obj_2_x, obj_1_x
                obj_1_y, obj_2_y = obj_2_y, obj_1_y
        dx = (obj_1_x - obj_2_x)
        dy = (obj_1_y - obj_2_y)
        if self.calibration_params['calibration_circles_xaxis']:
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
        if self.debug:
            self.cparams.print_()
        self.image.blur()
        if self.debug:
            self.image.show()
        self.image.mask()
        if self.debug:
            self.image.show()
        self.image.morph()
        if self.debug:
            self.image.show()

    def calibrate(self):
        """Determine coordinate conversion parameters."""
        if len(self.db.calibration_pixel_locations) > 1:
            calibration_circle_sep = float(
                self.calibration_params['calibration_circle_separation'])
            if self.calibration_params['calibration_circles_xaxis']:
                i = 0
            else:
                i = 1
            object_sep = abs(self.db.calibration_pixel_locations[0][i] -
                             self.db.calibration_pixel_locations[1][i])
            self.calibration_params['coord_scale'] = round(calibration_circle_sep
                                                           / object_sep, 4)

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
        camera_offset = np.array(
            self.calibration_params['camera_offset_coordinates'], dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        sign = [1 if s == 1 else -1 for s
                in self.calibration_params['image_bot_origin_location']]
        coord_scale = np.array([self.calibration_params['coord_scale'],
                                self.calibration_params['coord_scale']])
        db.coordinate_locations = []
        for o, object_pixel_location in enumerate(db.pixel_locations[:, :2]):
            radius = db.pixel_locations[:][o][2]
            moc = (camera_coordinates +
                   sign * coord_scale *
                   (self.calibration_params['center_pixel_location']
                    - object_pixel_location))
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
        camera_offset = np.array(
            self.calibration_params['camera_offset_coordinates'], dtype=float)
        camera_coordinates = coord + camera_offset  # image center coordinates
        center_pixel_location = self.calibration_params[
            'center_pixel_location'][:2]
        sign = [1 if s == 1 else -1 for s
                in self.calibration_params['image_bot_origin_location']]
        coord_scale = np.array([self.calibration_params['coord_scale'],
                                self.calibration_params['coord_scale']])
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
        total_rotation_angle = 0
        warning_issued = False
        for i in range(0, self.calibration_params['calibration_iters']):
            self.process_image()
            self.image.find(calibration=True)  # find objects
            # If not the last iteration, determine camera rotation angle
            if i != (self.calibration_params['calibration_iters'] - 1):
                # Check number of objects detected and notify user if needed.
                if len(self.db.calibration_pixel_locations) == 0:
                    print("ERROR: Calibration failed. No objects detected.")
                    sys.exit(0)
                if self.db.object_count > 2:
                    if not warning_issued:
                        print(" Warning: {} objects detected. "
                              "Exactly 2 reccomended. "
                              "Incorrect results likely.".format(
                                  self.db.object_count))
                        warning_issued = True
                if self.db.object_count < 2:
                    print(" ERROR: {} objects detected. "
                          "At least 2 required. Exactly 2 reccomended.".format(
                              self.db.object_count))
                    sys.exit(0)
                # Use detected objects to determine required rotation angle
                self.rotationdetermination()
                if abs(self.rotationangle) > 120:
                    print(" ERROR: Excessive rotation required. "
                          "Check that the calibration objects are parallel with "
                          "the desired axis and that they are the only two objects"
                          " detected.")
                    sys.exit(0)
                self.image.rotate_main_images(self.rotationangle)
                total_rotation_angle += self.rotationangle
        self.calibrate()
        self.db.print_coordinates()
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
        if self.params_loaded_from_json:
            self.image.image = self.image.marked
            self.image.save('calibration_result')

    def determine_coordinates(self):
        """Use calibration parameters to determine locations of objects."""
        self.image.rotate_main_images(self.calibration_params[
                                      'total_rotation_angle'])
        self.process_image()
        self.image.find(calibration=True)
        self.db.print_count(calibration=True)  # print detected objects count
        self.p2c(self.db)
        self.db.print_coordinates()
        if self.viewoutputimage:
            self.image.image = self.image.marked
            self.image.show()

if __name__ == "__main__":
    folder = os.path.dirname(os.path.realpath(__file__)) + os.sep
    print("Calibration image load...")
    P2C = Pixel2coord(DB(), calibration_image=folder +
                      "p2c_test_calibration.jpg")
    P2C.viewoutputimage = True
    # Calibration
    P2C.image.rotate_main_images(P2C.test_rotation)
    P2C.calibration()
    P2C.db.print_count(calibration=True)  # print number of objects detected
    if P2C.calibration_params['total_rotation_angle'] != 0:
        print(" Note: required rotation executed = {:.2f} degrees".format(
            P2C.calibration_params['total_rotation_angle']))
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
