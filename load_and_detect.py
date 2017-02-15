"""Load (recent image) and Detect commands to load as farmware."""

import os
import sys
import glob
from Plant_Detection import Plant_Detection

try:
    recent_image = max(glob.iglob('/tmp/images/*.[Jj][Pp][Gg]'),
                       key=os.path.getctime)
except ValueError:
    print("No images in /tmp/images")
    sys.exit(0)

PD = Plant_Detection(image=recent_image,
                     parameters_from_env_var=True, verbose=False)
PD.detect_plants()
