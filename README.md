# Plant Detection
Detects and marks green plants in a (not green) soil area image using Python OpenCV.

The goal is to mark unwanted volunteer plants for removal.

For an overview of the image processing performed, see [the wiki](../../wiki/Plant Detection Image Processing Steps).

## Contents
 * [Installation](#installation)
 * [Basic Usage](#basic-usage)
   * [Option 1: Run script](#run-the-script)
   * [Option 2: Python console](#alternatively-process-images-using-a-python-command-line)
   * [Option 3: GUI](#or-run-the-gui-and-move-the-sliders)
 * [Suggested Workflow](#image-file-processing-suggested-workflow)
 * [Tips](#tips)

---

## Installation

__Install OpenCV on Debian:__
```
sudo apt-get install python-opencv python-numpy python-picamera
```

## Basic Usage

#### Run the script:

Using the sample soil image, `soil_image.jpg`.

Run the script: `python Plant_Detection.py`

View `soil_image_marked.jpg`

#### Alternatively, process images using a python command line:
```python
from Plant_Detection import Plant_Detection
help(Plant_Detection)
PD = Plant_Detection(image='soil_image.jpg')
PD.calibrate()
PD.detect_plants()
PD = Plant_Detection(image='soil_image.jpg', morph=15, iterations=2, debug=True)
PD.detect_plants()
```

#### Or, run the GUI and move the sliders:
```python
python Plant_Detection_GUI.py
```
Default image to process is `soil_image.jpg`. To process other images, use:
```python
python Plant_Detection_GUI.py other_image_name.png
```
<img src="https://cloud.githubusercontent.com/assets/12681652/15620382/b7f31dd6-240e-11e6-853f-356d1a90376e.png" width="350">
<!--![plant detection gui screenshot](https://cloud.githubusercontent.com/assets/12681652/15620382/b7f31dd6-240e-11e6-853f-356d1a90376e.png)-->

## Image file processing suggested workflow

#### 1. Save image to be processed
For example: `test_image.jpg`

#### 2. Run the GUI and move the sliders:
```python
python Plant_Detection_GUI.py test_image.jpg
```
This will create a plant detection parameters input file from the slider values.

#### 3. Run detection:
```python
python Plant_Detection.py test_image.jpg
```
>Or, for more options, enter a python command line: `python`
```python
from Plant_Detection import Plant_Detection
PD = Plant_Detection(image='test_image.jpg', parameters_from_file=True)
PD.detect_plants()
```

#### 4. View output
Annotated image: `test_image_marked.png`

## Tips

#### View help
`python -c $'from Plant_Detection import Plant_Detection\nhelp(Plant_Detection)'`

#### Hue range aid
`python Plant_Detection_GUI.py pixel_to_coordinate/p2c_test_color.jpg`
