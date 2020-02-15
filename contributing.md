# Contributing

## Setup (Dependency installation)
```
pip install -r requirements.txt
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
`pip install prospector`
### Run
`python -m prospector`

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
`pip install coverage`
### Run
```
python -m coverage run -m unittest discover
python -m coverage html
```
open `coverage_html_report/index.html` in browser

## Pre-commit hooks
_Settings stored in `.pre-commit-config.yaml`_
### Setup
```
pip install pre-commit
python -m pre_commit install
```
### Run
```
git add .
python -m pre_commit run --all-files
git diff
```
