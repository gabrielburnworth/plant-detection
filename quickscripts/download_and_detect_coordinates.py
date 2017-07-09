"""Download an image from the Web App and detect coordinates.

download the image corresponding to the ID provided and run plant detection
and coordinate conversion
"""

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from plant_detection.PlantDetection import PlantDetection
from plant_detection import ENV

if __name__ == "__main__":
    IMAGE_ID = ENV.load('PLANT_DETECTION_selected_image', get_json=False)
    PD = PlantDetection(coordinates=True, app=True, app_image_id=IMAGE_ID)
    PD.detect_plants()
