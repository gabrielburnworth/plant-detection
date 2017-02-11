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
           parameters_from_json (boolean): load input parameters from json,
                overriding other parameter inputs, example:
                PLANT_DETECTION_options={"blur":15,"morph":8,"iterations":4,
                 "H":[37,82],"S":[38,255],"V":[61,255]}
                DB={"plants":[{"id":115,"device_id":76}]}
                (default = False)
           calibration_parameters_from_json (boolean): load calibration
                parameters from json overriding other parameter inputs
                see example in parameters_from_json
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
        self.known_plants = None  # default
        self.debug = False  # default
        self.save = True   # default
        self.parameters_from_file = False  # default
        self.parameters_from_json = False  # default
        self.calibration_parameters_from_json = False  # default
        self.params = Parameters()
        self.db = DB()
        self.capture = Capture().capture
        for key in kwargs:
            if key == 'image': self.image = kwargs[key]
            if key == 'coordinates': self.coordinates = kwargs[key]
            if key == 'calibration_img': self.calibration_img = kwargs[key]
            if key == 'known_plants': self.db.known_plants = kwargs[key]
            if key == 'debug': self.debug = kwargs[key]
            if key == 'blur': self.params.blur_amount = kwargs[key]
            if key == 'morph': self.params.morph_amount = kwargs[key]
            if key == 'iterations': self.params.iterations = kwargs[key]
            if key == 'array': self.params.array = kwargs[key]
            if key == 'save': self.save = kwargs[key]
            if key == 'clump_buster': self.params.clump_buster = kwargs[key]
            if key == 'HSV_min': self.params.HSV_min = kwargs[key]
            if key == 'HSV_max': self.params.HSV_max = kwargs[key]
            if key == 'parameters_from_file':
                self.parameters_from_file = kwargs[key]
            if key == 'parameters_from_json':
                self.parameters_from_json = kwargs[key]
            if key == 'calibration_parameters_from_json':
                self.calibration_parameters_from_json = kwargs[key]
        if self.calibration_img is not None:
            self.coordinates = True
        self.grey_out = False
        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
        self.output_text = False
        self.output_json = True
        self.input_parameters_filename = "plant-detection_inputs.txt"
        self.db.tmp_dir = None
        self.db.output_text = self.output_text
        self.final_debug_image = None

    def calibrate(self):
        """Initialize coordinate conversion module using calibration image."""
        if self.calibration_img is None and self.coordinates:
            # Calibration requested, but no image provided.
            # Take a calibration image.
            self.calibration_img = self.capture()
        if self.calibration_parameters_from_json:
            try:
                self.params.load_json()
                self.db.calibration_parameters = self.params
            except KeyError:
                print("JSON parameters load failed.")
        # Call coordinate conversion module
        P2C = Pixel2coord(self.db, calibration_image=self.calibration_img)
        P2C.calibration()  # calibrate and save values

    def detect_plants(self):
        """Detect the green objects in the image."""
        if self.parameters_from_file:
            # Requested to load detection parameters from file
            self.params.load(self.dir, self.input_parameters_filename)

        if self.parameters_from_json:
            # Requested to load detection parameters from json ENV variable
            try:
                self.params.load_json()
            except KeyError:
                print("JSON parameters load failed.")
            try:
                self.db.load_known_plants_from_json()
            except KeyError:
                print("JSON known plants load failed.")

        if self.output_text:
            self.params.print_()
            print("\nProcessing image: {}".format(self.image))

        if self.image is None:
            # No image provided. Capture one.
            self.image = Image(self.params, self.db) # create image object
            self.image.capture()
            self.image.save_annotated('photo')
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
        if self.params.clump_buster:
            self.image.clump_buster()

        # Optionally grey out regions not detected as objects
        if self.grey_out:
            self.image.grey()

        # Return coordinates if requested
        if self.coordinates:  # Convert pixel locations to coordinates
            P2C = Pixel2coord(self.db)  # Use calibration values created by calibrate()
            self.image.coordinates(P2C)  # get coordinates of all detected objects
            self.db.identify()  # organize objects into plants and weeds
            self.db.print_count()  # print number of objects detected
            if self.output_text:
                self.db.print_()  # print organized object data text to stdout
            if self.output_json:
                self.db.json_()  # print organized object data json to stdout
            if self.debug:
                self.image.save_annotated('contours')
                self.image.image = self.image.marked
                self.image.save_annotated('coordinates_found')
            self.image.label(P2C)  # mark each object with a colored circle
            self.image.grid(P2C)  # add coordinate grid and features
            self.image.save('marked')

        else:  # No coordinate conversion
            self.image.find()  # get pixel locations of objects
            if self.debug:
                self.image.save_annotated('contours')
            self.db.print_count()  # print number of objects detected
            if self.output_text:
                self.db.print_pixel()  # print object pixel location text
            self.image.image = self.image.marked  # Save marked soil image
            self.image.save('marked')

        if self.debug:
            self.final_debug_image = self.image.marked
            self.params.save(self.dir, self.input_parameters_filename)
            self.db.save_known_plants()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
        soil_image = directory + 'soil_image.jpg'
        PD = Plant_Detection(image=soil_image,
            blur=15, morph=6, iterations=4,
            calibration_img=directory + "PD/p2c_test_calibration.jpg",
            parameters_from_json=True,  # This overrides input in next line
            known_plants=[[200, 600, 100], [900, 200, 120]])
        PD.calibrate()  # use calibration img to get coordinate conversion data
        PD.detect_plants()  # detect coordinates and sizes of weeds and plants
    else:
        soil_image = sys.argv[1]
        PD = Plant_Detection(image=soil_image, parameters_from_file=True, debug=True)
        PD.detect_plants()
