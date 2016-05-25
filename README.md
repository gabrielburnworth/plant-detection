# plant-detection
Detects and marks green plants in a (not green) soil area image using Python OpenCV.

The goal is to mark unwanted volunteer plants for removal.

__Install OpenCV on Debian:__
```
sudo apt-get install python-opencv python-numpy
```
__Run the script:__

Download an image of soil with plants in it and save it as `soil_image.jpg` or use the sample.

Run the script: `python soil_image_plant_detection.py`

View `soil_image_marked.jpg`

__Alternatively, process images using a python command line:__
```python
from soil_image_plant_detection import detect_plants
help(detect_plants)
detect_plants('soil_image.jpg')
detect_plants('soil_image.jpg', morph=15, iterations=2, debug=True)
```

__Or, run the experimental GUI and move the sliders:__
```python
python Plant_Detection_GUI.py
```
