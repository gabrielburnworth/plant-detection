#!/usr/bin/env python
"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import sys, os
from PD.P2C import Pixel2coord
from PD.Image import Image
from PD.Parameters import Parameters
from PD.DB import DB
from PD.Capture import Capture

class Plant_Detection():
    """Detect plants in image and saves an image with plants marked.

       Kwargs:
           image (str): filename of image to process (default = None)
               None -> take photo instead
           coordinates (boolean): use coordinate conversion (default = False)
           calibration_img (filename): calibration image filename used to
               output coordinates instead of pixel locations (default = None)
           known_plants (list): {'x': x, 'y': y, 'radius': radius}
                                of known (intentional) plants
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
           parameters_from_env_var (boolean): load input parameters from
                environment variable, overriding other parameter inputs
                Example:
                PLANT_DETECTION_options={"blur":15,"morph":8,"iterations":4,
                 "H":[37,82],"S":[38,255],"V":[61,255]}
                DB={"plants":[{"id":115,"device_id":76}]}
                (default = False)
           calibration_parameters_from_env_var (boolean): load calibration
                parameters from environment variable,
                overriding other parameter inputs
                see example in parameters_from_env_var
                (default = False)

       Examples:
           PD = Plant_Detection()
           PD = Plant_Detection(image='soil_image.jpg', morph=3, iterations=10,
              debug=True)
           PD = Plant_Detection(image='soil_image.jpg', blur=9, morph=7,
              iterations=4, calibration_img="PD/p2c_test_calibration.jpg")
           PD = Plant_Detection(image='soil_image.jpg', blur=15,
              array=[[5, 'ellipse', 'erode',  2],
                     [3, 'ellipse', 'dilate', 8]], debug=True, save=False,
              clump_buster=True, HSV_min=[15, 15, 15], HSV_max=[85, 245, 245])
           PD.calibrate()
           PD.detect_plants()
    """
    def __init__(self, **kwargs):
        self.image = None
        self.coordinates = False
        self.calibration_img = None  # default
        self.debug = False  # default
        self.save = True   # default
        self.parameters_from_file = False  # default
        self.parameters_from_env_var = False  # default
        self.calibration_parameters_from_env_var = False  # default
        self.clump_buster = False  # default
        self.text_output = True  # default
        self.verbose = True  # default
        self.print_all_json = False  # default
        self.params = Parameters()
        self.db = DB()
        self.P2C = None
        self.capture = Capture().capture
        for key in kwargs:
            if key == 'image': self.image = kwargs[key]
            if key == 'coordinates': self.coordinates = kwargs[key]
            if key == 'calibration_img': self.calibration_img = kwargs[key]
            if key == 'known_plants': self.db.plants['known'] = kwargs[key]
            if key == 'debug': self.debug = kwargs[key]
            if key == 'blur': self.params.parameters['blur'] = kwargs[key]
            if key == 'morph': self.params.parameters['morph'] = kwargs[key]
            if key == 'iterations':
                self.params.parameters['iterations'] = kwargs[key]
            if key == 'array': self.params.array = kwargs[key]
            if key == 'save': self.save = kwargs[key]
            if key == 'clump_buster': self.clump_buster = kwargs[key]
            if key == 'HSV_min':
                HSV_min = kwargs[key]
                self.params.parameters['H'][0] = HSV_min[0]
                self.params.parameters['S'][0] = HSV_min[1]
                self.params.parameters['V'][0] = HSV_min[2]
            if key == 'HSV_max':
                HSV_max = kwargs[key]
                self.params.parameters['H'][1] = HSV_max[0]
                self.params.parameters['S'][1] = HSV_max[1]
                self.params.parameters['V'][1] = HSV_max[2]
            if key == 'parameters_from_file':
                self.parameters_from_file = kwargs[key]
            if key == 'parameters_from_env_var':
                self.parameters_from_env_var = kwargs[key]
            if key == 'calibration_parameters_from_env_var':
                self.calibration_parameters_from_env_var = kwargs[key]
            if key == 'text_output': self.text_output = kwargs[key]
            if key == 'verbose': self.verbose = kwargs[key]
            if key == 'print_all_json': self.print_all_json = kwargs[key]
        if self.calibration_img is not None:
            self.coordinates = True
        self.grey_out = False
        self.output_celeryscript = True
        self.db.tmp_dir = None
        self.db.output_text = self.verbose
        self.final_debug_image = None

    def calibrate(self):
        """Initialize coordinate conversion module using calibration image."""
        if self.calibration_img is None and self.coordinates:
            # Calibration requested, but no image provided.
            # Take a calibration image.
            self.calibration_img = self.capture()
        if self.calibration_parameters_from_env_var:
            try:
                self.params.load_env_var()
                self.db.calibration_parameters = self.params
            except KeyError:
                print("Environment variable parameters load failed.")
        # Call coordinate conversion module
        self.P2C = Pixel2coord(self.db, calibration_image=self.calibration_img)
        self.P2C.calibration()  # calibrate and save values
        if self.verbose and self.text_output:
            if self.P2C.calibration_params['total_rotation_angle'] != 0:
                print(" Note: required rotation executed = {:.2f} degrees".format(
                    self.P2C.calibration_params['total_rotation_angle']))
            self.db.print_count(calibration=True)  # print number of objects detected

    def detect_plants(self):
        """Detect the green objects in the image."""
        if self.parameters_from_file:
            # Requested to load detection parameters from file
            self.params.load()

        if self.parameters_from_env_var:
            # Requested to load detection parameters from json ENV variable
            try:
                self.params.load_env_var()
            except KeyError:
                print("Environment variable parameters load failed.")
            try:
                self.db.load_known_plants_from_env_var()
            except KeyError:
                print("Environment variable plants load failed.")

        if self.verbose and self.text_output:
            self.params.print_()
            print("\nProcessing image: {}".format(self.image))

        if self.image is None:
            # No image provided. Capture one.
            self.image = Image(self.params, self.db) # create image object
            self.image.capture()
            if self.debug:
                self.image.save('photo')
        else:
            # Image provided. Load it.
            filename = self.image
            self.image = Image(self.params, self.db) # create image object
            self.image.load(filename)

        # Blur image to simplify and reduce noise.
        self.image.blur()
        if self.debug:
            self.image.save_annotated('blurred')

        # Create a mask using the color range parameters
        self.image.mask()
        if self.debug:
            self.image.save_annotated('masked')
            self.image.mask2()

        # Transform mask to try to make objects more coherent
        self.image.morph()
        if self.debug:
            self.image.save_annotated('morphed')
            self.image.morph2()

        # Optionally break up masses by splitting them into quarters
        if self.clump_buster:
            self.image.clump_buster()

        # Optionally grey out regions not detected as objects
        if self.grey_out:
            self.image.grey()

        # Return coordinates if requested
        if self.coordinates:  # Convert pixel locations to coordinates
            self.P2C = Pixel2coord(self.db)  # Use saved calibration values
            self.image.coordinates(self.P2C)  # get coordinates of objects
            self.db.identify()  # organize objects into plants and weeds
            if self.text_output:
                self.db.print_count()  # print number of objects detected
            if self.verbose and self.text_output:
                self.db.print_()  # print organized object data text to stdout
            if self.output_celeryscript:
                self.db.output_CS()  # print object data JSON to stdout
            if self.debug:
                self.image.save_annotated('contours')
                self.image.image = self.image.marked
                self.image.save_annotated('coordinates_found')
            self.image.label(self.P2C)  # mark objects with colored circles
            self.image.grid(self.P2C)  # add coordinate grid and features
            self.image.save('marked')

        else:  # No coordinate conversion
            self.image.find()  # get pixel locations of objects
            self.image.label()  # Mark plants with red circle
            if self.debug:
                self.image.save_annotated('contours')
            if self.text_output:
                self.db.print_count()  # print number of objects detected
            if self.verbose and self.text_output:
                self.db.print_pixel()  # print object pixel location text
            self.image.image = self.image.marked  # Save marked soil image
            self.image.save('marked')

        if self.debug:
            self.final_debug_image = self.image.marked
            self.params.save()
            self.db.save_plants()

        if self.print_all_json:
            print("\nJSON:")
            print(self.params.parameters)
            print(self.db.plants)
            if self.P2C is not None: print(self.P2C.calibration_params)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
        soil_image = directory + 'soil_image.jpg'
        PD = Plant_Detection(image=soil_image,
            blur=15, morph=6, iterations=4,
            calibration_img=directory + "PD/p2c_test_calibration.jpg",
            known_plants=[{'x': 200, 'y': 600, 'radius': 100},
                          {'x': 900, 'y': 200, 'radius': 120}])
        PD.calibrate()  # use calibration img to get coordinate conversion data
        PD.detect_plants()  # detect coordinates and sizes of weeds and plants
    else:
        soil_image = sys.argv[1]
        PD = Plant_Detection(image=soil_image, parameters_from_file=True, debug=True)
        PD.detect_plants()
