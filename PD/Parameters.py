#!/usr/bin/env python
"""Parameters for Plant Detection.

For Plant Detection.
"""
import os
import cv2
import json
try:
    from .CeleryPy import CeleryPy
except:
    from CeleryPy import CeleryPy


class Parameters():
    """Input parameters for Plant Detection"""

    def __init__(self):
        self.parameters = {'blur': 5, 'morph': 5, 'iterations': 1,
                           'H': [30, 90], 'S': [20, 255], 'V': [20, 255]}
        self.array = None  # default
        self.kernel_type = 'ellipse'
        self.morph_type = 'close'
        self.parameters_from_file = False  # default
        self.dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
        self.input_parameters_file = "plant-detection_inputs.json"
        self.output_text = False
        self.output_json = False
        self.tmp_dir = None
        self.calibration_params_from_env_var = False

        # Create dictionaries of morph types
        self.kt = {}  # morph kernel type
        self.kt['ellipse'] = cv2.MORPH_ELLIPSE
        self.kt['rect'] = cv2.MORPH_RECT
        self.kt['cross'] = cv2.MORPH_CROSS
        self.mt = {}  # morph type
        self.mt['close'] = cv2.MORPH_CLOSE
        self.mt['open'] = cv2.MORPH_OPEN

    def save(self):
        """Save input parameters to file"""
        def save(directory):
            with open(directory + self.input_parameters_file, 'w') as f:
                json.dump(self.parameters, f)
        try:
            save(self.dir)
        except IOError:
            self.tmp_dir = "/tmp/"
            save(self.tmp_dir)

    def save_to_env_var(self):
        """Save input parameters to environment variable"""
        CeleryPy().save_inputs_to_env_var(self.parameters)

    def load(self):
        """Load input parameters from file"""
        def load(directory):
            with open(directory + self.input_parameters_file, 'r') as f:
                self.parameters = json.load(f)
        try:
            try:
                load(self.dir)
            except IOError:
                self.tmp_dir = "/tmp/"
                load(self.tmp_dir)
        except IOError:
            pass

    def load_env_var(self):
        """Read input parameters from JSON in environment variable"""
        try:
            self.parameters = json.loads(os.environ['PLANT_DETECTION_options'])
        except KeyError:
            # Load defaults for environment variable
            self.parameters = {'blur': 15, 'morph': 6, 'iterations': 4,
                               'H': [30, 90], 'S': [50, 255], 'V': [50, 255]}

    def print_(self):
        """Print input parameters"""
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

if __name__ == "__main__":
    parameters = Parameters()
    parameters.load()
    parameters.print_()
    parameters.parameters['iterations'] = 4
    parameters.print_()
    parameters.save()
