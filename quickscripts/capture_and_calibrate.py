"""Capture and Calibrate commands to load as farmware."""

from Plant_Detection import Plant_Detection

if __name__ == "__main__":
    PD = Plant_Detection(coordinates=True, app=True)
    PD.calibrate()
