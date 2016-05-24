# plant-detection
Detects and marks green plants in a (not green) soil area image using Python OpenCV.

The goal is to mark unwanted volunteer plants for removal.

Install OpenCV on Debian:
```sudo apt-get install python-opencv python-numpy```

Download an image of soil with plants in it and save it as ```soil_image.jpg```

Run the script:
```python soil_image_plant_detection.py```

View ```soil_image_marked.jpg```

Alternatively, process images using a python command line:
```from soil_image_plant_detection import detect_plants
help(detect_plants)
detect_plants('soil_image.jpg', morph=5, iterations=4, debug=True)```
