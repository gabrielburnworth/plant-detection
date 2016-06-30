# pixel-to-coordinate
Image Pixel Location to Machine Coordinate Conversion using OpenCV

__Install OpenCV on Debian:__
```
sudo apt-get install python-opencv python-numpy
```
__Run the script:__

Run the script: `python pixel2coord.py`

__Alternatively, use a python command line:__
```python
from pixel2coord import calibration, determine_coordinates
determine_coordinates("p2c_test_objects.jpg", *calibration("p2c_test_calibration.jpg"))
```
