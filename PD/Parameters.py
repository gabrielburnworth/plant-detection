#!/usr/bin/env python
"""Parameters for Plant Detection.

For Plant Detection.
"""
import os
import json
import cv2
import CeleryPy


class Parameters(object):
    """Input parameters for Plant Detection."""

    def __init__(self):
        """Set initial attributes and defaults."""
        self.parameters = {'blur': 5, 'morph': 5, 'iterations': 1,
                           'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.array = None  # default
        self.kernel_type = 'ellipse'
        self.morph_type = 'close'
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.input_parameters_file = "plant-detection_inputs.json"
        self.output_text = False
        self.output_json = False
        self.tmp_dir = None
        self.json_input_parameters = None
        self.calibration_data = None
        self.env_var_name = 'PLANT_DETECTION_options'

        # Create dictionaries of morph types
        self.cv2_kt = {}  # morph kernel type
        self.cv2_kt['ellipse'] = cv2.MORPH_ELLIPSE
        self.cv2_kt['rect'] = cv2.MORPH_RECT
        self.cv2_kt['cross'] = cv2.MORPH_CROSS
        self.cv2_mt = {}  # morph type
        self.cv2_mt['close'] = cv2.MORPH_CLOSE
        self.cv2_mt['open'] = cv2.MORPH_OPEN
        self.cv2_mt['erode'] = 'erode'
        self.cv2_mt['dilate'] = 'dilate'

    def save(self):
        """Save input parameters to file."""
        def _save(directory):
            input_filename = directory + self.input_parameters_file
            with open(input_filename, 'w') as input_file:
                json.dump(self.parameters, input_file)
        try:
            _save(self.dir)
        except IOError:
            self.tmp_dir = "/tmp/"
            _save(self.tmp_dir)

    def save_to_env_var(self):
        """Save input parameters to environment variable."""
        self.json_input_parameters = CeleryPy.set_user_env(
            self.env_var_name,
            json.dumps(self.parameters))
        os.environ[self.env_var_name] = json.dumps(self.parameters)

    def load(self):
        """Load input parameters from file."""
        def _load(directory):
            input_filename = directory + self.input_parameters_file
            with open(input_filename, 'r') as input_file:
                self.parameters = json.load(input_file)
        try:
            _load(self.dir)
        except IOError:
            self.tmp_dir = "/tmp/"
            _load(self.tmp_dir)

    def load_env_var(self):
        """Read input parameters from JSON in environment variable."""
        self.parameters = json.loads(os.environ[self.env_var_name])

    def load_defaults_for_env_var(self):
        """Load default input parameters for environment variable."""
        self.parameters = {'blur': 15, 'morph': 6, 'iterations': 4,
                           'H': [30, 90], 'S': [50, 255], 'V': [50, 255]}

    def print_input(self):
        """Print input parameters."""
        print('Processing Parameters:')
        print('-' * 25)
        print('Blur kernel size: {}'.format(self.parameters['blur']))
        print('Morph kernel size: {}'.format(self.parameters['morph']))
        print('Iterations: {}'.format(self.parameters['iterations']))
        print('Hue:\n\tMIN: {}\n\tMAX: {}'.format(*self.parameters['H']))
        print('Saturation:\n\tMIN: {}\n\tMAX: {}'.format(
            *self.parameters['S']))
        print('Value:\n\tMIN: {}\n\tMAX: {}'.format(*self.parameters['V']))
        print('-' * 25)
