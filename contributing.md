# Contributing

## Setup
#### python / OpenCV 3:
`pip install -r requirements.txt`
#### python / OpenCV 2 (Debian):
`sudo apt-get install python-opencv python-numpy python-requests python-redis`
#### python3 / OpenCV 3 (Debian):
```
sudo apt-get install python3-numpy python3-requests python3-redis
pip install opencv-python
```

## Support
| Python | NumPy | OpenCV |
|:------:|:-----:|:------:|
| 2.7    | 1.8   | 2.4    |
| 3.5    | 1.12  | 3.2    |
#### Check versions:
```
python --version
python -c 'import cv2; print("OpenCV " + cv2.__version__)'
python -c 'import numpy; print("NumPy " + numpy.__version__)'
```
_Build matrix in `.travis.yml`_

## Style
_Settings are stored in `.landscape.yml`_
### Setup
`sudo pip install prospector`
### Run
`prospector`

## Test Suite
_Can also be run via `python tests/tests.py`_
### Setup
`pip install fakeredis`
### Run
`python -m unittest discover -v`
#### Manual Tests
```
python Plant_Detection.py --GUI
python -m PD.Capture
python -m PD.P2C
```

## Coverage
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
