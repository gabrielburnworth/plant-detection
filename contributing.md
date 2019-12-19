# Contributing

## Setup (Dependency installation)
#### python / OpenCV 3:
`pip install -r requirements.txt`
#### python / OpenCV 2 (Debian):
`sudo apt-get install python-opencv python-numpy python-requests`
#### python3 / OpenCV 3/4 (Debian):
```
sudo apt-get install python3-numpy python3-requests
pip install opencv-python
```

## Supported dependency versions
|             | Python | NumPy | OpenCV |
|:-----------:|:------:|:-----:|:------:|
| **Legacy**  | 2.7    | 1.8   | 2.4    |
| **Current** | 3.8    | 1.17  | 3.4    |
#### Check versions:
```
python --version
python -c 'import cv2; print("OpenCV " + cv2.__version__)'
python -c 'import numpy; print("NumPy " + numpy.__version__)'
```
_Build matrix in `.travis.yml`_

## Static code analysis
_Settings are stored in `.landscape.yml`_
### Setup
`sudo pip install prospector`
### Run
`prospector`

## Test Suite
_Can also be run via `python -m plant_detection.tests.tests`_
### Setup
`pip install fakeredis`
### Run
`python -m unittest discover -v`
#### Manual Tests
```
python -m plant_detection.PlantDetection --GUI
python -m plant_detection.Capture
python -m plant_detection.P2C
```

## Test coverage
_Settings stored in `.coveragerc`_
### Setup
`sudo pip install coverage`
### Run
```
coverage run -m unittest discover
coverage html
```
open `coverage_html_report/index.html` in browser

## Pre-commit hooks
_Settings stored in `.pre-commit-config.yaml`_
### Setup
```
pip install pre-commit
pre-commit install
```
### Run
```
git add .
pre-commit run --all-files
git diff
```
