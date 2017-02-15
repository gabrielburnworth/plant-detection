"""Capture and Detect commands to load as farmware."""

from Plant_Detection import Plant_Detection

PD = Plant_Detection(parameters_from_env_var=True, verbose=False)
PD.detect_plants()
