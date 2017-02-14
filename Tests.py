#!/usr/bin/env python
"""Plant Detection Test Suite

For Plant Detection.
"""
import os
import json
import unittest
import cv2
from Plant_Detection import Plant_Detection

class PDTestJSONinput(unittest.TestCase):
    """Test ENV VAR inputs"""
    def setUp(self):
        self.pd = Plant_Detection(image='soil_image.jpg',
                                  parameters_from_env_var=True,
                                  text_output=False)
        self.json_params = {"blur": 15, "morph": 6, "iterations": 4,
                            "H": [30, 90], "S": [20, 255], "V": [20, 255]}
        self.json_db = {"device_id": 1,
                            "plants": [{"x": 100, "y": 200, "radius": 300},
                                       {"x": 400, "y": 500, "radius": 600}]}
        self.json_known_plants = self.json_db['plants']
        os.environ["PLANT_DETECTION_options"] = json.dumps(self.json_params)
        os.environ["DB"] = json.dumps(self.json_db)
        self.pd.detect_plants()

    def test_json_parameters_input(self):
        """Load JSON input parameters from ENV VAR"""
        self.assertEqual(self.pd.db.plants['known'], self.json_known_plants)
        self.assertEqual(self.pd.params.parameters, self.json_params)

class PDTestNoJSONinput(unittest.TestCase):
    """Test defaults"""
    def setUp(self):
        self.parameters = {'blur': 5, 'morph': 5, 'iterations': 1,
                          'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.pd = Plant_Detection(image='soil_image.jpg', text_output=False)
        self.pd.detect_plants()

    def test_json_parameters_input(self):
        """Do not load JSON input parameters from ENV VAR"""
        self.assertEqual(self.pd.db.plants['known'], [])
        self.assertEqual(self.pd.params.parameters, self.parameters)

class PDTestCalibration(unittest.TestCase):
    """Test calibration process"""
    def setUp(self):
        self.pd = Plant_Detection(image="PD/p2c_test_objects.jpg",
            calibration_img="PD/p2c_test_calibration.jpg",
            HSV_min=[160, 100, 100], HSV_max=[20, 255, 255],
            morph=15, blur=5, text_output=False)
        self.pd.calibrate()
        self.calibration_json = {"blur": 5, "morph": 15, "calibration_iters": 3,
          "H": [160, 20], "S": [100, 255], "V": [100, 255],
          "calibration_circles_xaxis": True,
          "camera_offset_coordinates": [200, 100],
          "image_bot_origin_location": [0, 1],
          "calibration_circle_separation": 1000,
          "total_rotation_angle": 0.0,
           "coord_scale": 1.7182,
           "center_pixel_location": [465, 290]}
        self.objects = [{'y': 300.69, 'x': 300.0, 'radius': 46.86},
                        {'y': 599.66, 'x': 897.94, 'radius': 46.86},
                        {'y': 800.68, 'x': 98.97, 'radius': 47.53}]
        if cv2.__version__[0] == '3':
            self.calibration_json['total_rotation_angle'] = 0.014
            self.objects = [{'y': 300.69, 'x': 300.0, 'radius': 45.6},
                            {'y': 599.66, 'x': 897.94, 'radius': 45.57},
                            {'y': 800.68, 'x': 98.97, 'radius': 46.28}]

    def test_calibration_inputs(self):
        """Check calibration input parameters"""
        self.assertEqual(
            self.pd.P2C.calibration_params['calibration_iters'],
            self.calibration_json['calibration_iters'])
        self.assertEqual(
            self.pd.P2C.calibration_params['calibration_circle_separation'],
            self.calibration_json['calibration_circle_separation'])
        self.assertEqual(
            self.pd.P2C.calibration_params['blur'],
            self.calibration_json['blur'])
        self.assertEqual(
            self.pd.P2C.calibration_params['morph'],
            self.calibration_json['morph'])
        self.assertEqual(
            self.pd.P2C.calibration_params['H'],
            self.calibration_json['H'])
        self.assertEqual(
            self.pd.P2C.calibration_params['S'],
            self.calibration_json['S'])
        self.assertEqual(
            self.pd.P2C.calibration_params['V'],
            self.calibration_json['V'])
        self.assertEqual(
            self.pd.P2C.calibration_params['calibration_circles_xaxis'],
            self.calibration_json['calibration_circles_xaxis'])
        self.assertEqual(
            self.pd.P2C.calibration_params['camera_offset_coordinates'],
            self.calibration_json['camera_offset_coordinates'])
        self.assertEqual(
            self.pd.P2C.calibration_params['image_bot_origin_location'],
            self.calibration_json['image_bot_origin_location'])


    def test_calibration_results(self):
        """Check calibration results"""
        self.assertEqual(self.pd.P2C.calibration_params['total_rotation_angle'],
                         self.calibration_json['total_rotation_angle'])
        self.assertEqual(self.pd.P2C.calibration_params['coord_scale'],
                         self.calibration_json['coord_scale'])
        self.assertEqual(self.pd.P2C.calibration_params['center_pixel_location'],
                         self.calibration_json['center_pixel_location'])

    def test_object_coordinate_detection(self):
        """Determine coordinates of test objects"""
        self.pd.detect_plants()
        self.assertEqual(self.pd.db.plants['remove'], self.objects)

class PDTestArgs(unittest.TestCase):
    """Test plant detection input arguments"""
    def setUp(self):
        self.image = 'soil_image.jpg'
        self.coordinates = True
        self.calibration_img = "PD/p2c_test_calibration.jpg"
        self.known_plants = [[200, 600, 100], [900, 200, 120]]
        self.blur = 9; self.morph = 7; self.iterations = 3
        self.array = [[5, 'ellipse', 'erode',  2],
               [3, 'ellipse', 'dilate', 8]]
        self.debug = True; self.save = False; self.clump_buster = True
        self.HSV_min = [15, 15, 15]; self.HSV_max = [85, 245, 245]
        self.parameters_from_file = True
        self.parameters_from_env_var = True
        self.calibration_parameters_from_env_var = True
        self.default_input_params = {'blur': 5, 'morph': 5, 'iterations': 1,
            'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.set_input_params = {'blur': 9, 'morph': 7, 'iterations': 3,
            'H': [15, 85], 'S': [15, 245], 'V': [15, 245]}

    def test_input_args(self):
        """Set all arguments"""
        pd = Plant_Detection(
            image=self.image,
            coordinates=self.coordinates,
            calibration_img=self.calibration_img,
            known_plants=self.known_plants,
            blur=self.blur, morph=self.morph, iterations=self.iterations,
            array=self.array,
            debug=self.debug, save=self.save, clump_buster=self.clump_buster,
            HSV_min=self.HSV_min, HSV_max=self.HSV_max,
            parameters_from_file=self.parameters_from_file,
            parameters_from_env_var=self.parameters_from_env_var,
            calibration_parameters_from_env_var=self.calibration_parameters_from_env_var)
        self.assertEqual(pd.image, self.image)
        self.assertEqual(pd.coordinates, self.coordinates)
        self.assertEqual(pd.calibration_img, self.calibration_img)
        self.assertEqual(pd.db.plants['known'], self.known_plants)
        self.assertEqual(pd.params.parameters, self.set_input_params)
        self.assertEqual(pd.params.array, self.array)
        self.assertEqual(pd.debug, self.debug)
        self.assertEqual(pd.save, self.save)
        self.assertEqual(pd.clump_buster, self.clump_buster)
        self.assertEqual(pd.parameters_from_file,
                         self.parameters_from_file)
        self.assertEqual(pd.parameters_from_env_var,
                         self.parameters_from_env_var)
        self.assertEqual(pd.calibration_parameters_from_env_var,
                         self.calibration_parameters_from_env_var)

    def test_input_defaults(self):
        """Use defaults"""
        pd = Plant_Detection()
        self.assertEqual(pd.image, None)
        self.assertEqual(pd.coordinates, False)
        self.assertEqual(pd.calibration_img, None)
        self.assertEqual(pd.db.plants['known'], [])
        self.assertEqual(pd.params.parameters, self.default_input_params)
        self.assertEqual(pd.params.array, None)
        self.assertEqual(pd.debug, False)
        self.assertEqual(pd.save, True)
        self.assertEqual(pd.clump_buster, False)
        self.assertEqual(pd.parameters_from_file, False)
        self.assertEqual(pd.parameters_from_env_var, False)
        self.assertEqual(pd.calibration_parameters_from_env_var, False)

class PDTestOutput(unittest.TestCase):
    """Test plant detection results"""
    def setUp(self):
        # self.maxDiff = None
        self.pd = Plant_Detection(image="soil_image.jpg",
                             calibration_img = "PD/p2c_test_calibration.jpg",
                             known_plants=[{'x': 200, 'y': 600, 'radius': 100},
                                           {'x': 900, 'y': 200, 'radius': 120}],
                             blur=15, morph=6, iterations=4,
                             text_output=False)
        self.pd.detect_plants()
        self.input_params = {'blur': 15, 'morph': 6, 'iterations': 4,
                             'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.calibration = {'blur': 5, 'morph': 15, 'calibration_iters': 3,
                           'H': [160, 20], 'S': [100, 255], 'V': [100, 255],
                           'calibration_circles_xaxis': True,
                           'camera_offset_coordinates': [200, 100],
                           'image_bot_origin_location': [0, 1],
                           'calibration_circle_separation': 1000,
                           'total_rotation_angle': 0.0,
                           'coord_scale': 1.7182,
                           'center_pixel_location': [465, 290]}
        self.plants = {
            'known': [{'y': 600, 'x': 200, 'radius': 100},
                      {'y': 200, 'x': 900, 'radius': 120}],
            'save': [{'y': 85.91, 'x': 837.8, 'radius': 80.52},
                     {'y': 189.01, 'x': 901.37, 'radius': 65.32},
                     {'y': 579.04, 'x': 236.43, 'radius': 91.23}],
            'remove': [{'y': 41.24, 'x': 1428.86, 'radius': 73.59},
                       {'y': 42.96, 'x': 607.56, 'radius': 82.26},
                       {'y': 103.1, 'x': 1260.48, 'radius': 3.44},
                       {'y': 152.92, 'x': 1214.09, 'radius': 62.0},
                       {'y': 216.5, 'x': 1373.88, 'radius': 13.82},
                       {'y': 231.96, 'x': 1286.25, 'radius': 61.8},
                       {'y': 285.23, 'x': 1368.72, 'radius': 14.4},
                       {'y': 412.37, 'x': 1038.83, 'radius': 73.97},
                       {'y': 479.38, 'x': 1531.95, 'radius': 80.96},
                       {'y': 500.0, 'x': 765.64, 'radius': 80.02},
                       {'y': 608.25, 'x': 1308.59, 'radius': 148.73},
                       {'y': 676.97, 'x': 59.46, 'radius': 60.95},
                       {'y': 914.09, 'x': 62.89, 'radius': 82.37}]
            }
        if cv2.__version__[0] == '3':
            self.calibration['total_rotation_angle'] = 0.014
            self.plants = {
                'known': [{'y': 600, 'x': 200, 'radius': 100},
                          {'y': 200, 'x': 900, 'radius': 120}],
                'save': [{'y': 85.91, 'x': 837.8, 'radius': 78.18},
                         {'y': 189.01, 'x': 901.37, 'radius': 63.42},
                         {'y': 579.04, 'x': 236.43, 'radius': 88.57}],
                'remove': [{'y': 39.52, 'x': 1427.14, 'radius': 72.01},
                           {'y': 41.24, 'x': 607.56, 'radius': 80.32},
                           {'y': 103.1, 'x': 1260.48, 'radius': 2.86},
                           {'y': 152.92, 'x': 1214.09, 'radius': 60.53},
                           {'y': 216.5, 'x': 1373.88, 'radius': 13.42},
                           {'y': 231.96, 'x': 1286.25, 'radius': 60.0},
                           {'y': 285.23, 'x': 1368.72, 'radius': 13.98},
                           {'y': 412.37, 'x': 1038.83, 'radius': 72.26},
                           {'y': 479.38, 'x': 1533.67, 'radius': 79.36},
                           {'y': 500.0, 'x': 765.64, 'radius': 77.69},
                           {'y': 608.25, 'x': 1308.59, 'radius': 146.17},
                           {'y': 676.97, 'x': 57.74, 'radius': 59.95},
                           {'y': 914.09, 'x': 61.17, 'radius': 80.64}]
                }

    def test_output(self):
        """Check detect plants results"""
        self.assertEqual(self.pd.db.plants, self.plants)
        self.assertEqual(self.pd.params.parameters, self.input_params)
        self.assertEqual(self.pd.P2C.calibration_params, self.calibration)

if __name__ == '__main__':
    unittest.main()
