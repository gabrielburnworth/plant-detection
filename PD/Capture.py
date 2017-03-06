#!/usr/bin/env python
"""Plant Detection Image Capture.

For Plant Detection.
"""
import sys
import os
import platform
from time import sleep
import cv2
try:
    import redis
except ImportError:
    pass


USE_RPI_CAMERA = False
USING_RPI = False

if platform.uname()[4].startswith("arm") and USE_RPI_CAMERA:
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    USING_RPI = True


class Capture(object):
    """Capture image for Plant Detection."""

    def __init__(self, r=None):
        """Set initial attributes."""
        self.image = None
        self.ret = None
        self.camera_port = None
        self.test_coordinates = [600, 400, 0]
        self.image_captured = False
        self.silent = False
        self.redis = r

    def getcoordinates(self):
        """Get machine coordinates from bot."""
        try:  # return bot coordintes
            if self.redis is not None:
                _redis = self.redis
            else:
                _redis = redis.StrictRedis()
            bot_x = int(_redis.lindex('BOT_STATUS.location', 0))
            bot_y = int(_redis.lindex('BOT_STATUS.location', 1))
            bot_z = int(_redis.lindex('BOT_STATUS.location', 2))
            return [bot_x, bot_y, bot_z]
        except:  # noqa pylint:disable=W0702
            return self.test_coordinates  # return testing coordintes

    def camera_check(self):
        """Check for camera at ports 0 and 1."""
        if not os.path.exists('/dev/video' + str(self.camera_port)):
            if not self.silent:
                print("No camera detected at video{}.".format(
                    self.camera_port))
            self.camera_port = 1
            if not self.silent:
                print("Trying video{}...".format(self.camera_port))
            if not os.path.exists('/dev/video' + str(self.camera_port)):
                if not self.silent:
                    print("No camera detected at video{}.".format(
                        self.camera_port))

    def capture(self):
        """Take a photo."""
        if USING_RPI and USE_RPI_CAMERA:
            # With Raspberry Pi Camera:
            with PiCamera() as camera:
                camera.resolution = (1920, 1088)
                raw_capture = PiRGBArray(camera)
                sleep(0.1)
                camera.capture(raw_capture, format="bgr")
                self.image = raw_capture.array
        else:
            self.camera_port = 0
            # With USB cameras:
            # image_width = 1600
            # image_height = 1200
            discard_frames = 20
            self.camera_check()  # check for camera
            camera = cv2.VideoCapture(self.camera_port)
            sleep(0.1)
            # try:
            #     camera.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
            #     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
            # except AttributeError:
            #     camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, image_width)
            #     camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, image_height)
            for _ in range(discard_frames):
                camera.grab()
            self.ret, self.image = camera.read()
            camera.release()
        if not self.ret:
            print("Problem getting image.")
            sys.exit(0)
        self.image_captured = True
        return self.image


if __name__ == "__main__":
    DIR = os.path.dirname(os.path.realpath(__file__))[:-3] + os.sep
    IMG = Capture().capture()
    cv2.imwrite(DIR + 'capture.jpg', IMG)
