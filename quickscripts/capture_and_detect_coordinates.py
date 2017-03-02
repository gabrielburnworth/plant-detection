"""Capture and Detect commands to load as farmware.

take a photo and run plant detection
and coordinate conversion
"""
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from Plant_Detection import Plant_Detection

if __name__ == "__main__":
    PD = Plant_Detection(coordinates=True, app=True)
    PD.detect_plants()
