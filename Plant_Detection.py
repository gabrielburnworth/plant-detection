#!/usr/bin/env python
"""Plant Detection.

Detects green plants on a dirt background
 and marks them with red circles.
"""
import sys
import os
from PD.P2C import Pixel2coord
from PD.Image import Image
from PD.Parameters import Parameters
from PD.DB import DB
from PD.Capture import Capture


class Plant_Detection():
    """Detect plants in image and output an image with plants marked.

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
               [morph kernel size, morph kernel type, morph type, iterations]
               example:
               [{"size": 5, "kernel": 'ellipse', "type": 'erode',  "iters": 2},
                {"size": 3, "kernel": 'ellipse', "type": 'dilate', "iters": 8}]
                      (default = None)
           save (boolean): save images (default = True)
           clump_buster (boolean): attempt to break
                                   plant clusters (default = False)
           HSV_min (list): green lower bound Hue(0-179), Saturation(0-255),
                           and Value(0-255) (default = [30, 20, 20])
           HSV_max (list): green upper bound Hue(0-179), Saturation(0-255),
                           and Value(0-255) (default = [90, 255, 255])
           from_file (boolean): load data from file
                plant-detection_inputs.json
                plant-detection_p2c_calibration_parameters.json
                plant-detection_plants.json
                (default = False)
           from_env_var (boolean): load data from environment variable,
                overriding other parameter inputs
                Example:
                PLANT_DETECTION_options={"blur": 15, "morph": 8, "iterations": 4,
                 "H": [37, 82], "S": [38, 255], "V": [61, 255]}
                DB={"plants": [{"x": 10, "y": 20, "radius": 30}]}
                PLANT_DETECTION_calibration={'total_rotation_angle': 0.0,
                                             'coord_scale': 1.7182,
                                             'center_pixel_location': [465, 290]}
                (default = False)
           text_output (boolean): print text to STDOUT (default = True)
           verbose (boolean): print verbose text to STDOUT.
                otherwise, print condensed text output (default = True)
           print_all_json (boolean): print all JSON data used to STDOUT
                (defalut = False)
           grey_out (boolean): grey out regions in image that have
                not been selected (default = False)
           draw_contours (boolean): draw an outline around the boundary of
                detected plants (default = True)
           circle_plants (boolean): draw an enclosing circle around
                detected plants (default = True)
           GUI (boolean): settings for the local GUI (default = False)
           app (boolean): connect to the FarmBot web app (default = False)

       Examples:
           PD = Plant_Detection()
           PD.detect_plants()

           PD = Plant_Detection(image='soil_image.jpg', morph=3, iterations=10,
              debug=True)
           PD.detect_plants()

           PD = Plant_Detection(image='soil_image.jpg', blur=9, morph=7,
              iterations=4, calibration_img="PD/p2c_test_calibration.jpg")
           PD.calibrate()
           PD.detect_plants()

           PD = Plant_Detection(image='soil_image.jpg', blur=15, grey_out=True,
              array=[{"size": 5, "kernel": 'ellipse', "type": 'erode',  "iters": 2},
                     {"size": 3, "kernel": 'ellipse', "type": 'dilate', "iters": 8}],
                     debug=True, clump_buster=False,
                     HSV_min=[30, 15, 15], HSV_max=[85, 245, 245])
           PD.detect_plants()
    """

    def __init__(self, **kwargs):
        # Initialization
        self.image = None
        self.params = Parameters()
        self.db = DB()
        self.P2C = None
        self.capture = Capture().capture
        self.final_marked_image = None
        self.output_celeryscript_points = False
        self.db.tmp_dir = None

        # Arguments
        # Defaults
        self.coordinates = False
        self.calibration_img = None
        self.debug = False
        self.save = True
        self.clump_buster = False
        self.from_file = False
        self.from_env_var = False
        self.text_output = True
        self.verbose = True
        self.print_all_json = False
        self.grey_out = False
        self.draw_contours = True
        self.circle_plants = True
        self.GUI = False
        self.app = False
        # Inputs
        for key in kwargs:
            if key == 'image':
                self.image = kwargs[key]
            if key == 'coordinates':
                self.coordinates = kwargs[key]
            if key == 'calibration_img':
                self.calibration_img = kwargs[key]
            if key == 'known_plants':
                self.db.plants['known'] = kwargs[key]
            if key == 'debug':
                self.debug = kwargs[key]
            if key == 'blur':
                self.params.parameters['blur'] = kwargs[key]
            if key == 'morph':
                self.params.parameters['morph'] = kwargs[key]
            if key == 'iterations':
                self.params.parameters['iterations'] = kwargs[key]
            if key == 'array':
                self.params.array = kwargs[key]
            if key == 'save':
                self.save = kwargs[key]
            if key == 'clump_buster':
                self.clump_buster = kwargs[key]
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
            if key == 'from_file':
                self.from_file = kwargs[key]
            if key == 'from_env_var':
                self.from_env_var = kwargs[key]
            if key == 'text_output':
                self.text_output = kwargs[key]
            if key == 'verbose':
                self.verbose = kwargs[key]
            if key == 'print_all_json':
                self.print_all_json = kwargs[key]
            if key == 'grey_out':
                self.grey_out = kwargs[key]
            if key == 'draw_contours':
                self.draw_contours = kwargs[key]
            if key == 'circle_plants':
                self.circle_plants = kwargs[key]
            if key == 'GUI':
                self.GUI = kwargs[key]
            if key == 'app':
                self.app = kwargs[key]
        # Changes based on inputs
        if self.calibration_img is not None:
            self.coordinates = True
        self.db.output_text = self.verbose
        if self.GUI:
            self.save = False
            self.text_output = False
        if self.app:
            self.verbose = False
            self.from_env_var = True

    def calibrate(self):
        """Initialize coordinate conversion module using calibration image."""
        if self.calibration_img is None and self.coordinates:
            # Calibration requested, but no image provided.
            # Take a calibration image.
            self.calibration_img = self.capture()

        # Set calibration input parameters
        if self.from_env_var:
            try:
                self.params.load_env_var()
                calibration_input = self.params.parameters.copy()
            except (KeyError, ValueError):
                print("Warning: Environment variable calibration "
                      "parameters load failed.")
                calibration_input = None
        elif self.from_file:  # try to load from file
            try:
                self.params.load()
                calibration_input = self.params.parameters.copy()
            except IOError:
                print("Warning: Calibration data file load failed. Using defaults.")
                calibration_input = None
        else:  # Use default calibration inputs
            calibration_input = None

        # Call coordinate conversion module
        self.P2C = Pixel2coord(self.db,
                               calibration_image=self.calibration_img,
                               calibration_data=calibration_input)
        # Perform calibration
        self.P2C.calibration()
        if self.save:
            self.P2C.image.image = self.P2C.image.marked
            self.P2C.image.save('calibration_result')

        # Print verbose results
        if self.verbose and self.text_output:
            if self.P2C.calibration_params['total_rotation_angle'] != 0:
                print(" Note: required rotation executed = {:.2f} degrees".format(
                    self.P2C.calibration_params['total_rotation_angle']))
            if self.debug:
                # print number of objects detected
                self.db.print_count(calibration=True)
                # print coordinate locations of calibration objects
                self.P2C.p2c(self.db)
                self.db.print_coordinates()
                print('')

        # Print condensed output if verbose output is not chosen
        if self.text_output and not self.verbose:
            print("Calibration complete. (rotation:{}, scale:{})".format(
                self.P2C.calibration_params['total_rotation_angle'],
                self.P2C.calibration_params['coord_scale']))

        # Save calibration data
        if self.from_env_var:
            # to environment variable
            self.P2C.save_calibration_data_to_env_var()
        elif self.from_file:  # to file
            self.P2C.save_calibration_parameters()
        else:  # to Parameters() instance
            self.params.calibration_data = self.P2C.calibration_params

    def detect_plants(self):
        """Detect the green objects in the image."""

        # Load input parameters
        if self.from_file:
            # Requested to load detection parameters from file
            try:
                self.params.load()
            except IOError:
                print("Warning: Input parameter file load failed. Using defaults.")
        if self.app:
            self.db.load_plants_from_web_app()
        if self.from_env_var:
            # Requested to load detection parameters from json ENV variable
            try:
                self.params.load_env_var()
            except (KeyError, ValueError):
                print("Warning: Environment variable parameters load failed.")
                self.params.load_defaults_for_env_var()

        # Print input parameters and filename of image to process
        if self.verbose and self.text_output:
            self.params.print_input()
            print("\nProcessing image: {}".format(self.image))

        # Get image to process
        if self.image is None:
            # No image provided. Capture one.
            self.image = Image(self.params, self.db)  # create image object
            self.image.capture()
            if self.debug:
                self.image.save('photo')
        else:  # Image provided. Load it.
            filename = self.image
            self.image = Image(self.params, self.db)  # create image object
            self.image.load(filename)
        self.image.debug = self.debug

        # Process image in preparation for detecting plants (blur, mask, morph)
        self.image.initial_processing()

        # Optionally break up masses by splitting them into quarters
        if self.clump_buster:
            self.image.clump_buster()

        # Optionally grey out regions not detected as objects
        if self.grey_out:
            self.image.grey()

        # Return coordinates if requested
        if self.coordinates:  # Convert pixel locations to coordinates
            # Load calibration data
            if self.from_env_var:
                calibration_data = 'env_var'
            elif self.from_file:
                calibration_data = 'file'
            else:  # use data saved in self.params
                calibration_data = self.params.calibration_data
            # Initialize coordinate conversion module
            self.P2C = Pixel2coord(self.db, calibration_data=calibration_data)
            # Check for coordinate conversion calibration results
            try:
                self.P2C.calibration_params['coord_scale']
            except KeyError:
                print("ERROR: Coordinate conversion calibration values "
                      "not found. Run calibration first.")
                sys.exit(0)
            # Determine object coordinates
            self.image.coordinates(self.P2C)
            # Organize objects into plants and weeds
            self.db.identify()
            if self.db.plants['safe_remove']:
                self.image.safe_remove(self.P2C)
            # Print and output results
            if self.text_output:
                self.db.print_count()  # print number of objects detected
            if self.verbose and self.text_output:
                self.db.print_identified()  # print organized object data text to stdout
            if self.output_celeryscript_points:
                self.db.output_CS()  # print object data JSON to stdout
            if self.app:
                self.db.upload_weeds()  # add weeds to FarmBot Farm Designer
            if self.debug:
                self.image.save_annotated('contours')
                self.image.image = self.image.marked
                self.image.save_annotated('coordinates_found')
            if self.circle_plants:
                self.image.label(self.P2C)  # mark objects with colored circles
            self.image.grid(self.P2C)  # add coordinate grid and features

        else:  # No coordinate conversion
            # get pixel locations of objects
            self.image.find(draw_contours=self.draw_contours)
            if self.circle_plants:
                self.image.label()  # Mark plants with red circle
            if self.debug:
                self.image.save_annotated('contours')
            if self.text_output:
                self.db.print_count()  # print number of objects detected
            if self.verbose and self.text_output:
                self.db.print_pixel()  # print object pixel location text
            self.image.image = self.image.marked

        # Final marked image
        if self.save or self.debug:
            self.image.save('marked')
        elif self.GUI:
            self.final_marked_image = self.image.marked

        # Print raw JSON to STDOUT
        if self.print_all_json:
            print("\nJSON:")
            print(self.params.parameters)
            print(self.db.plants)
            if self.P2C is not None:
                print(self.P2C.calibration_params)

        # Print condensed inputs if verbose output is not chosen
        if self.text_output and not self.verbose:
            print('{}: {}'.format('known plants input',
                                  self.db.plants['known']))
            print('{}: {}'.format('parameters input',
                                  self.params.parameters))
            print('{}: {}'.format('coordinates input',
                                  Capture().getcoordinates()))

        # Save input parameters
        if self.from_env_var:
            # to environment variable
            self.params.save_to_env_var()
        elif self.save:
            # to file
            self.params.save()
        elif self.GUI:
            # to file for GUI
            self.params.save()

        # Save plants
        if self.save:
            self.db.save_plants()

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
        PD = Plant_Detection(
            image=soil_image, from_file=True, debug=True)
        PD.detect_plants()
