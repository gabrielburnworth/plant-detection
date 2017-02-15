"""Capture and Calibrate commands to load as farmware."""

import os
from Plant_Detection import Plant_Detection

directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
soil_image = directory + 'soil_image.jpg'
PD = Plant_Detection(image=soil_image, coordinates=True,
                     calibration_parameters_from_env_var=True, verbose=False)
PD.calibrate()
