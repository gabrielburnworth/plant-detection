"""Download an image from the Web App and detect coordinates.

download the image corresponding to the ID provided and run plant detection
and coordinate conversion
"""

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from PlantDetection import PlantDetection

if __name__ == "__main__":
    PD = PlantDetection(coordinates=True, app=True)
    PD.detect_plants()
