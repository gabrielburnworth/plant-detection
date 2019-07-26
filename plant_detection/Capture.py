#!/usr/bin/env python
"""Plant Detection Image Capture.

For Plant Detection.
"""
import sys
import os
from time import time, sleep
from subprocess import call
import cv2
from plant_detection import ENV
from plant_detection.Log import log

CAMERA = ENV.load('camera', get_json=False)
if CAMERA is None:
    CAMERA = 'USB'  # default camera
else:
    if 'RPI' in CAMERA:
        CAMERA = 'RPI'  # Raspberry Pi Camera
    else:
        CAMERA = 'USB'


class Capture(object):
    """Capture image for Plant Detection."""

    def __init__(self):
        """Set initial attributes."""
        self.image = None
        self.ret = None
        self.camera_port = None
        self.image_captured = False
        self.silent = False

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
                    log("USB Camera not detected.",
                        message_type='error', title='take-photo')

    def save(self, filename_only=False, add_timestamp=True):
        """Save captured image."""
        directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
        try:
            testfilename = directory + 'test_write.try_to_write'
            testfile = open(testfilename, "w")
            testfile.close()
            os.remove(testfilename)
        except IOError:
            directory = '/tmp/images/'
        if add_timestamp:
            image_filename = directory + 'capture_{timestamp}.jpg'.format(
                timestamp=int(time()))
        else:
            image_filename = directory + 'capture.jpg'
        if not filename_only:
            cv2.imwrite(image_filename, self.image)
        return image_filename

    def capture(self):
        """Take a photo."""
        if CAMERA == 'RPI':
            # With Raspberry Pi Camera:
            try:
                retcode = call(["raspistill", "-w", "640", "-h", "480",
                                "-o", self.save(filename_only=True)])
            except OSError:
                log("Raspberry Pi Camera not detected.",
                    message_type='error', title='take-photo')
                sys.exit(0)
            else:
                if retcode == 0:
                    print("Image saved: {}".format(
                        self.save(filename_only=True)))
                    return self.save(filename_only=True)
                else:
                    log("Problem getting image.",
                        message_type='error', title='take-photo')
                    sys.exit(0)
        else:  # With USB camera:
            self.camera_port = 0
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
                log("Problem getting image.",
                    message_type='error', title='take-photo')
                sys.exit(0)
            self.image_captured = True
            return self.save()


if __name__ == "__main__":
    Capture().capture()
