#!/usr/bin/env python
"""Parameters for Plant Detection.

For Plant Detection.
"""
import sys, os
import cv2
import json

class Parameters():
    def __init__(self, **kwargs):
        self.blur_amount = 5  # default
        self.morph_amount = 5  # default
        self.iterations = 1  # default
        self.array = None  # default
        self.clump_buster = False  # default
        self.HSV_min = [30, 20, 20]  # default
        self.HSV_max = [90, 255, 255]  # default
        self.kernel_type = 'ellipse'
        self.morph_type = 'close'
        self.parameters_from_file = False  # default
        self.output_text = False
        self.output_json = False
        self.tmp_dir = None

        # Create dictionaries of morph types
        self.kt = {}  # morph kernel type
        self.kt['ellipse'] = cv2.MORPH_ELLIPSE
        self.kt['rect'] = cv2.MORPH_RECT
        self.kt['cross'] = cv2.MORPH_CROSS
        self.mt = {}  # morph type
        self.mt['close'] = cv2.MORPH_CLOSE
        self.mt['open'] = cv2.MORPH_OPEN

    def save(self, directory, filename):
        try:
            with open(directory + filename, 'w') as f:
                f.write('blur_amount {}\n'.format(self.blur_amount))
                f.write('morph_amount {}\n'.format(self.morph_amount))
                f.write('iterations {}\n'.format(self.iterations))
                f.write('clump_buster {}\n'.format(
                    [1 if self.clump_buster else 0][0]))
                f.write('HSV_min {:d} {:d} {:d}\n'.format(*self.HSV_min))
                f.write('HSV_max {:d} {:d} {:d}\n'.format(*self.HSV_max))
        except IOError:
            self.tmp_dir = "/tmp/"
            self.save(self.tmp_dir, filename)

    def load(self, directory, filename):
        def load(directory):  # Load input parameters from file
            with open(directory + filename, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                if "blur_amount" in line:
                    self.blur_amount = int(line[1])
                    if self.blur_amount % 2 == 0:
                        self.blur_amount += 1
                if "morph_amount" in line:
                    self.morph_amount = int(line[1])
                if "iterations" in line:
                    self.iterations = int(line[1])
                if "clump_buster" in line:
                    self.clump_buster = int(line[1])
                if "HSV_min" in line:
                    self.HSV_min = [int(line[1]),
                                    int(line[2]),
                                    int(line[3])]
                if "HSV_max" in line:
                    self.HSV_max = [int(line[1]),
                                    int(line[2]),
                                    int(line[3])]
        try:
            try:
                load(directory)
            except IOError:
                self.tmp_dir = "/tmp/"
                load(self.tmp_dir)
        except IOError:
            pass

    def load_json(self):
        try:
            params_json = json.loads(os.environ['PLANT_DETECTION_options'])
            # Read inputs from env vars
            self.HSV_min = [params_json['H'][0], params_json['S'][0], params_json['V'][0]]
            self.HSV_max = [params_json['H'][1], params_json['S'][1], params_json['V'][1]]
            self.blur_amount = int(params_json['blur'])
            self.morph_amount = int(params_json['morph'])
            self.iterations = int(params_json['iterations'])
        except KeyError:
            pass

    def print_(self):
        print('Processing Parameters:')
        print('-' * 25)
        print('Blur kernel size: {}'.format(self.blur_amount))
        print('Morph kernel size: {}'.format(self.morph_amount))
        print('Iterations: {}'.format(self.iterations))
        print('Clump Buster: {}'.format(self.clump_buster))
        print('Hue:\n\tMIN: {}\n\tMAX: {}'.format(
              self.HSV_min[0], self.HSV_max[0]))
        print('Saturation:\n\tMIN: {}\n\tMAX: {}'.format(
              self.HSV_min[1], self.HSV_max[1]))
        print('Value:\n\tMIN: {}\n\tMAX: {}'.format(
              self.HSV_min[2], self.HSV_max[2]))
        print('-' * 25)

if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
    filename = "plant-detection_inputs.txt"
    parameters = Parameters()
    parameters.load(dir, filename)
    parameters.print_()
    parameters.iterations = 4
    parameters.print_()
    parameters.save(dir, filename)
