#!/usr/bin/env python
"""Plant Detection Tests

For Plant Detection.
"""
import os
import sys
import json
import unittest
import numpy as np
from Plant_Detection import Plant_Detection


def assert_dict_values_almost_equal(assertAE, object1, object2):
    def shape(objects):
        if isinstance(objects, dict):
            formatted_objects = objects
        else:
            formatted_objects = {}
            formatted_objects['this'] = objects
        return formatted_objects
    f_object1 = shape(object1)
    f_object2 = shape(object2)
    for (_, dicts1), (_, dicts2) in zip(f_object1.items(), f_object2.items()):
        for dict1, dict2 in zip(dicts1, dicts2):
            for (_, value1), (_, value2) in zip(dict1.items(), dict2.items()):
                assertAE(value1, value2, delta=5)


def subset(dictionary, keylist):
    dict_excerpt = {key: dictionary[key] for key in keylist}
    return dict_excerpt


def compare_calibration_results(self):
    self.assertAlmostEqual(self.calibration['total_rotation_angle'],
                           self.pd.P2C.calibration_params[
                           'total_rotation_angle'], places=1)
    self.assertAlmostEqual(self.calibration['coord_scale'],
                           self.pd.P2C.calibration_params[
                           'coord_scale'], places=3)
    self.assertEqual(self.calibration['center_pixel_location'],
                     self.pd.P2C.calibration_params['center_pixel_location'])


def check_file_length(self, expected_length):
    self.outfile.close()
    self.outfile = open('text_output_test.txt', 'r')
    self.assertEqual(sum(1 for line in self.outfile), expected_length)
    self.outfile.close()
    self.outfile = open('text_output_test.txt', 'w')


def get_average_pixel_value(image):
    pixel_mean = round(np.mean(image), 1)
    return pixel_mean


class PDTestJSONinput(unittest.TestCase):
    """Test ENV VAR inputs"""

    def setUp(self):
        self.pd = Plant_Detection(image='soil_image.jpg',
                                  from_env_var=True,
                                  text_output=False)
        self.json_params = {"blur": 15, "morph": 6, "iterations": 4,
                            "H": [30, 90], "S": [20, 255], "V": [20, 255]}
        os.environ["PLANT_DETECTION_options"] = json.dumps(self.json_params)
        self.pd.detect_plants()

    def test_json_parameters_input(self):
        """Load JSON input parameters from ENV VAR"""
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

    def test_calibration_inputs(self):
        """Check calibration input parameters"""
        calibration_input_keys = ["blur", "morph", "calibration_iters",
                                  "H", "S", "V",
                                  "calibration_circles_xaxis",
                                  "camera_offset_coordinates",
                                  "image_bot_origin_location",
                                  "calibration_circle_separation"]
        self.assertEqual(
            subset(self.calibration_json, calibration_input_keys),
            subset(self.pd.P2C.calibration_params, calibration_input_keys))

    def test_calibration_results(self):
        """Check calibration results"""
        calibration_results_keys = ["total_rotation_angle",
                                    "coord_scale",
                                    "center_pixel_location"]
        static_results = subset(self.calibration_json,
                                calibration_results_keys)
        test_results = subset(self.pd.P2C.calibration_params,
                              calibration_results_keys)
        self.assertAlmostEqual(static_results['total_rotation_angle'],
                               test_results['total_rotation_angle'], places=1)
        self.assertAlmostEqual(static_results['coord_scale'],
                               test_results['coord_scale'], places=3)
        self.assertEqual(static_results['center_pixel_location'],
                         test_results['center_pixel_location'])

    def test_object_coordinate_detection(self):
        """Determine coordinates of test objects"""
        self.pd.detect_plants()
        assert_dict_values_almost_equal(self.assertAlmostEqual,
                                        self.pd.db.plants['remove'],
                                        self.objects)


class PDTestArgs(unittest.TestCase):
    """Test plant detection input arguments"""

    def setUp(self):
        self.image = 'soil_image.jpg'
        self.coordinates = True
        self.calibration_img = "PD/p2c_test_calibration.jpg"
        self.known_plants = [[200, 600, 100], [900, 200, 120]]
        self.blur = 9
        self.morph = 7
        self.iterations = 3
        self.array = [[5, 'ellipse', 'erode',  2],
                      [3, 'ellipse', 'dilate', 8]]
        self.debug = True
        self.save = False
        self.clump_buster = True
        self.HSV_min = [15, 15, 15]
        self.HSV_max = [85, 245, 245]
        self.from_file = True
        self.from_env_var = True
        self.default_input_params = {'blur': 5, 'morph': 5, 'iterations': 1,
                                     'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.set_input_params = {'blur': 9, 'morph': 7, 'iterations': 3,
                                 'H': [15, 85], 'S': [15, 245], 'V': [15, 245]}
        self.text_output = False
        self.verbose = False
        self.print_all_json = True
        self.grey_out = True
        self.draw_contours = False
        self.circle_plants = False
        self.GUI = True
        self.app = True

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
            from_file=self.from_file,
            from_env_var=self.from_env_var,
            text_output=self.text_output,
            verbose=self.verbose,
            print_all_json=self.print_all_json,
            grey_out=self.grey_out,
            draw_contours=self.draw_contours,
            circle_plants=self.circle_plants,
            GUI=self.GUI, app=self.app
        )
        self.assertEqual(pd.image, self.image)
        self.assertEqual(pd.coordinates, self.coordinates)
        self.assertEqual(pd.calibration_img, self.calibration_img)
        self.assertEqual(pd.db.plants['known'], self.known_plants)
        self.assertEqual(pd.params.parameters, self.set_input_params)
        self.assertEqual(pd.params.array, self.array)
        self.assertEqual(pd.debug, self.debug)
        self.assertEqual(pd.save, self.save)
        self.assertEqual(pd.clump_buster, self.clump_buster)
        self.assertEqual(pd.from_file, self.from_file)
        self.assertEqual(pd.from_env_var, self.from_env_var)
        self.assertEqual(pd.text_output, self.text_output)
        self.assertEqual(pd.verbose, self.verbose)
        self.assertEqual(pd.print_all_json, self.print_all_json)
        self.assertEqual(pd.grey_out, self.grey_out)
        self.assertEqual(pd.draw_contours, self.draw_contours)
        self.assertEqual(pd.circle_plants, self.circle_plants)
        self.assertEqual(pd.GUI, self.GUI)
        self.assertEqual(pd.app, self.app)

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
        self.assertEqual(pd.from_file, False)
        self.assertEqual(pd.from_env_var, False)
        self.assertEqual(pd.text_output, True)
        self.assertEqual(pd.verbose, True)
        self.assertEqual(pd.print_all_json, False)
        self.assertEqual(pd.grey_out, False)
        self.assertEqual(pd.draw_contours, True)
        self.assertEqual(pd.circle_plants, True)
        self.assertEqual(pd.GUI, False)
        self.assertEqual(pd.app, False)


class PDTestOutput(unittest.TestCase):
    """Test plant detection results"""

    def setUp(self):
        # self.maxDiff = None
        self.pd = Plant_Detection(image="soil_image.jpg",
                                  calibration_img="PD/p2c_test_calibration.jpg",
                                  known_plants=[{'x': 200, 'y': 600, 'radius': 100},
                                                {'x': 900, 'y': 200, 'radius': 120}],
                                  blur=15, morph=6, iterations=4,
                                  text_output=False)
        self.pd.calibrate()
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
        self.object_count = 16
        self.plants = {
            'known': [{'y': 600, 'x': 200, 'radius': 100},
                      {'y': 200, 'x': 900, 'radius': 120}],
            'save': [{'y': 189.01, 'x': 901.37, 'radius': 65.32},
                     {'y': 579.04, 'x': 236.43, 'radius': 91.23}],
            'safe_remove': [{'y': 85.91, 'x': 837.8, 'radius': 80.52}],
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
                       {'y': 914.09, 'x': 62.89, 'radius': 82.37},
                       {'y': 84.2, 'x': 772.51, 'radius': 20.06}]
        }

    def test_output(self):
        """Check detect plants results"""
        # self.maxDiff = None
        # self.assertEqual(self.pd.db.plants, self.plants)
        assert_dict_values_almost_equal(self.assertAlmostEqual,
                                        self.pd.db.plants,
                                        self.plants)
        self.assertEqual(self.pd.params.parameters, self.input_params)
        compare_calibration_results(self)

    def test_object_count(self):
        """Check for correct object count"""
        self.assertEqual(self.pd.db.object_count, self.object_count)


class ENV_VAR(unittest.TestCase):
    """Test environment variable use"""

    def setUp(self):
        self.input_params = {'blur': 15, 'morph': 6, 'iterations': 4,
                             'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.input_plants = {'plants': [{'y': 600, 'x': 200, 'radius': 100},
                                        {'y': 200, 'x': 900, 'radius': 120}]}
        self.calibration_input_params = {'blur': 5, 'morph': 15,
                                         'H': [160, 20], 'S': [100, 255], 'V': [100, 255]}
        self.calibration = {'total_rotation_angle': 0.0,
                            'coord_scale': 1.7182,
                            'center_pixel_location': [465, 290]}

    def test_set_inputs(self):
        """Set input environment variable"""
        os.environ["PLANT_DETECTION_options"] = json.dumps(self.input_params)
        os.environ["DB"] = json.dumps(self.input_plants)
        pd = Plant_Detection(image="soil_image.jpg",
                             from_env_var=True,
                             text_output=False, save=False)
        pd.detect_plants()
        self.assertEqual(pd.params.parameters,
                         self.input_params)

    def test_calibration_ENV_VAR(self):
        """Use calibration data environment variable"""
        os.environ["PLANT_DETECTION_options"] = json.dumps(
            self.calibration_input_params)
        os.environ["DB"] = json.dumps(self.input_plants)
        self.pd = Plant_Detection(calibration_img="PD/p2c_test_calibration.jpg",
                                  from_env_var=True,
                                  text_output=False, save=False)
        self.pd.calibrate()
        compare_calibration_results(self)

        os.environ["PLANT_DETECTION_options"] = json.dumps(self.input_params)
        pd = Plant_Detection(image="soil_image.jpg",
                             from_env_var=True,
                             text_output=False, save=False)
        pd.detect_plants()


class TestFromFile(unittest.TestCase):
    """Test file use"""

    def setUp(self):
        self.pd = Plant_Detection(
            calibration_img="PD/p2c_test_calibration.jpg",
            text_output=False, save=False)
        self.pd.calibrate()
        self.pd.P2C.save_calibration_parameters()
        self.object_count = 16

    def test_detect_coordinates(self):
        """Detect coordinates, getting calibration parameters from file"""
        pd = Plant_Detection(image="soil_image.jpg",
                             from_file=True, coordinates=True,
                             text_output=False, save=False)
        pd.detect_plants()
        self.assertEqual(pd.db.object_count, self.object_count)


class PDTestArray(unittest.TestCase):
    """Test input parameter array use"""

    def setUp(self):
        self.pd = Plant_Detection(
            image="soil_image.jpg",
            array=[{"size": 5, "kernel": 'ellipse', "type": 'erode',  "iters": 2},
                   {"size": 3, "kernel": 'ellipse', "type": 'dilate', "iters": 8}],
            text_output=False, save=False)
        self.pd.detect_plants()
        self.object_count = 29

    def test_detect(self):
        """Detect plants using array input"""
        self.assertEqual(self.pd.db.object_count, self.object_count)


class PDTestClumpBuster(unittest.TestCase):
    """Test other options"""

    def setUp(self):
        """Test clump buster"""
        self.pd = Plant_Detection(
            image="soil_image.jpg",
            morph=10,
            text_output=False, save=False, clump_buster=True)
        self.pd.detect_plants()
        self.object_count = 39

    def test_clump_buster(self):
        """Test clump buster"""
        self.assertEqual(self.pd.db.object_count, self.object_count)


class PDTestGreyOut(unittest.TestCase):
    """Test grey out option"""

    def setUp(self):
        pd = Plant_Detection(image="soil_image.jpg",
                             text_output=False, save=False,
                             grey_out=True)
        pd.detect_plants()
        self.pixel_mean = get_average_pixel_value(pd.image.image)
        self.expected_pixel_mean = 138.7

    def test_grey_out(self):
        """Test grey out option"""
        self.assertAlmostEqual(self.pixel_mean,
                               self.expected_pixel_mean,
                               delta=0.25)


class PDTestCirclePlants(unittest.TestCase):
    """Test circle plants option"""

    def setUp(self):
        pd = Plant_Detection(image="soil_image.jpg",
                             text_output=False, save=False,
                             draw_contours=False,
                             circle_plants=True)
        pd.detect_plants()
        self.pixel_mean = get_average_pixel_value(pd.image.image)
        self.expected_pixel_mean = 69.9

    def test_circle_plants(self):
        """Test circle plants option"""
        self.assertAlmostEqual(self.pixel_mean,
                               self.expected_pixel_mean,
                               delta=0.25)


class PDTestDrawContours(unittest.TestCase):
    """Test draw contours option"""

    def setUp(self):
        pd = Plant_Detection(image="soil_image.jpg",
                             text_output=False, save=False,
                             draw_contours=True,
                             circle_plants=False)
        pd.detect_plants()
        self.pixel_mean = get_average_pixel_value(pd.image.image)
        self.expected_pixel_mean = 72.9

    def test_draw_contours(self):
        """Test draw contours option"""
        self.assertAlmostEqual(self.pixel_mean,
                               self.expected_pixel_mean,
                               delta=0.25)


class PDTestTextOutput(unittest.TestCase):
    """Test all text output"""

    def setUp(self):
        self.outfile = open('text_output_test.txt', 'w')
        sys.stdout = self.outfile

    def test_verbose_text_output_no_coordinates(self):
        """Test verbpse text output without coordinate conversion"""
        pd = Plant_Detection(
            image="soil_image.jpg",
            save=False, print_all_json=True)
        self.outfile.truncate()
        pd.detect_plants()
        check_file_length(self, 71)

    def test_condensed_text_output_no_coordinates(self):
        """Test condensed text output without coordinate conversion"""
        pd = Plant_Detection(
            image="soil_image.jpg",
            verbose=False,
            save=False, print_all_json=True)
        self.outfile.truncate()
        pd.detect_plants()
        check_file_length(self, 8)

    def test_verbose_text_output(self):
        """Test verbpse text output"""
        pd = Plant_Detection(
            image="soil_image.jpg",
            calibration_img="PD/p2c_test_calibration.jpg",
            save=False, print_all_json=True)
        self.outfile.truncate()
        pd.calibrate()
        pd.detect_plants()
        check_file_length(self, 80)

    def test_condensed_text_output(self):
        """Test condensed text output"""
        pd = Plant_Detection(
            image="soil_image.jpg",
            calibration_img="PD/p2c_test_calibration.jpg",
            verbose=False,
            save=False, print_all_json=True)
        self.outfile.truncate()
        pd.calibrate()
        pd.detect_plants()
        check_file_length(self, 10)

    def tearDown(self):
        self.outfile.close()
        sys.stdout = sys.__stdout__


class PDTestDebugMode(unittest.TestCase):
    """Test debug option"""

    def test_debug_no_coordinates(self):
        """Test debug mode without coordinate conversion"""
        pd = Plant_Detection(image="soil_image.jpg",
                             text_output=False, save=False,
                             debug=True)
        pd.detect_plants()
        self.assertTrue(os.path.exists('soil_image_masked2.jpg'))

    def test_debug_with_coordinates(self):
        """Test debug mode with coordinate conversion"""
        pd = Plant_Detection(
            image="soil_image.jpg",
            calibration_img="PD/p2c_test_calibration.jpg",
            text_output=False, save=False,
            debug=True)
        pd.calibrate()
        pd.detect_plants()
        self.assertTrue(os.path.exists('soil_image_coordinates_found.jpg'))
        os.remove('soil_image_masked2.jpg')
