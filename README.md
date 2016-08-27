# plant-detection
Detects and marks green plants in a (not green) soil area image using Python OpenCV.

The goal is to mark unwanted volunteer plants for removal.

__Install OpenCV on Debian:__
```
sudo apt-get install python-opencv python-numpy python-picamera
```
__Run the script:__

Download an image of soil with plants in it and save it as `soil_image.jpg` or use the sample.

Run the script: `python soil_image_plant_detection.py`

View `soil_image_marked.jpg`

__Alternatively, process images using a python command line:__
```python
from soil_image_plant_detection import Detect_plants
help(Detect_plants)
DP = Detect_plants(image='soil_image.jpg')
DP.calibrate()
DP.detect_plants()
DP = Detect_plants(image='soil_image.jpg', morph=15, iterations=2, debug=True)
DP.detect_plants()
```

__Or, run the GUI and move the sliders:__
```python
python Plant_Detection_GUI.py
```
![plant detection gui screenshot](https://cloud.githubusercontent.com/assets/12681652/15620382/b7f31dd6-240e-11e6-853f-356d1a90376e.png)
