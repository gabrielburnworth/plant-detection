#!/usr/bin/env python
"""Plant Detection Image Capture.

For Plant Detection.
"""
import sys, os
import cv2
import platform
from time import sleep
from datetime import datetime
try:
    import gi
    gi.require_version('GExiv2', '0.10')
    from gi.repository import GExiv2
    exif_import = True
except ImportError:
    exif_import = False

use_rpi_camera = False
using_rpi = False

if platform.uname()[4].startswith("arm") and use_rpi_camera:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    using_rpi = True

class Capture():
    """Capture image for Plant Detection"""
    def __init__(self):
        self.image = None
        self.ret = None
        self.camera_port = 0
        self.timestamp = datetime.now().isoformat()
        self.current_coordinates = None
        self.test_coordinates = [600, 400]
        self.output_text = True

    def _getcoordinates(self):
        """Get machine coordinates from bot."""
        # For now, return testing coordintes:
        return self.test_coordinates

    def capture(self):
        """Take a photo."""
        if using_rpi and use_rpi_camera:
            # With Raspberry Pi Camera:
            with PiCamera() as camera:
                camera.resolution = (1920, 1088)
                rawCapture = PiRGBArray(camera)
                sleep(0.1)
                camera.capture(rawCapture, format="bgr")
                self.image = rawCapture.array
        else:
            # With USB cameras:
            camera_port = 0
            image_width = 1600
            image_height = 1200
            discard_frames = 20
            camera = cv2.VideoCapture(camera_port)
            sleep(0.1)
            try:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
            except AttributeError:
                camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, image_width)
                camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, image_height)
            for _ in range(discard_frames):
                camera.grab()
            self.ret, self.image = camera.read()
            camera.release()
        if not self.ret:
            print("No camera detected.")
            sys.exit(0)
        return self.image

if __name__ == "__main__":
    directory = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
    from Image import Image
    from Parameters import Parameters
    from DB import DB
    image = Capture().capture()
    wimage = Image(Parameters(), DB())
    wimage.image = image
    wimage.save("capture")

    if exif_import:
        exif = GExiv2.Metadata(directory + 'capture.jpg')
        current_coordinates = Capture()._getcoordinates()
        timestamp = Capture().timestamp
        exif['Exif.Image.ImageDescription'] = 'Coordinates: {}, Timestamp: {}'.format(
                                              current_coordinates, timestamp)
        print(exif.get_comment())
        exif.save_file()
