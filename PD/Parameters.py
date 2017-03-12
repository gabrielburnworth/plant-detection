#!/usr/bin/env python
"""Parameters for Plant Detection.

For Plant Detection.
"""
import os
import json
import cv2
from PD import ENV


class Parameters(object):
    """Input parameters for Plant Detection."""

    def __init__(self):
        """Set initial attributes and defaults."""
        self.parameters = {'blur': 5, 'morph': 5, 'iterations': 1,
                           'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.defaults = {'blur': 15, 'morph': 6, 'iterations': 4,
                         'H': [30, 90], 'S': [50, 255], 'V': [50, 255]}
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
        self.json_input_parameters = ENV.save(self.env_var_name,
                                              self.parameters)

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
        self._add_missing()
        return ""

    def load_env_var(self):
        """Read input parameters from JSON in environment variable."""
        self.parameters = ENV.load(self.env_var_name)
        if not isinstance(self.parameters, dict):
            self.load_defaults_for_env_var()
            message = "Warning: Environment variable parameters load failed."
        else:
            self._add_missing()
            message = ""
        return message

    def _add_missing(self):
        for key, value in self.defaults.items():
            if key not in self.parameters:
                self.parameters[key] = value

    def load_defaults_for_env_var(self):
        """Load default input parameters for environment variable."""
        self.parameters = self.defaults

    def print_input(self):
        """Print input parameters."""
        print('Processing Parameters:')
        print('-' * 25)
        if self.array is None:
            print('Blur kernel size: {}'.format(self.parameters['blur']))
            print('Morph kernel size: {}'.format(self.parameters['morph']))
            print('Iterations: {}'.format(self.parameters['iterations']))
        else:
            print('List of morph operations performed:')
            for number, morph in enumerate(self.array):
                print('{indent}Morph operation {number}'.format(
                    indent=' ' * 2, number=number + 1))
                for key, value in morph.items():
                    print('{indent}{morph_property}: {morph_value}'.format(
                        indent=' ' * 4, morph_property=key, morph_value=value))
        print('Hue:\n\tMIN: {}\n\tMAX: {}'.format(*self.parameters['H']))
        print('Saturation:\n\tMIN: {}\n\tMAX: {}'.format(
            *self.parameters['S']))
        print('Value:\n\tMIN: {}\n\tMAX: {}'.format(*self.parameters['V']))
        print('-' * 25)
