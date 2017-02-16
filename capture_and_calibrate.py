"""Capture and Calibrate commands to load as farmware."""

from Plant_Detection import Plant_Detection

PD = Plant_Detection(coordinates=True,
                     calibration_parameters_from_env_var=True, verbose=False)
PD.calibrate()
