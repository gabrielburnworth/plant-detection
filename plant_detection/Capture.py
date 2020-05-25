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

CAMERA = (ENV.load('camera', get_json=False) or 'USB').upper()


class Capture(object):
    """Capture image for Plant Detection."""

    def __init__(self, directory=None):
        """Set initial attributes."""
        self.image = None
        self.ret = None
        self.camera_port = None
        self.image_captured = False
        self.silent = False
        self.directory = directory

    def camera_check(self):
        """Check for camera at ports 0 and 1."""
        if not os.path.exists('/dev/video' + str(self.camera_port)):
            if not self.silent:
                print('No camera detected at video{}.'.format(
                    self.camera_port))
            self.camera_port = 1
            if not self.silent:
                print('Trying video{}...'.format(self.camera_port))
            if not os.path.exists('/dev/video' + str(self.camera_port)):
                if not self.silent:
                    print('No camera detected at video{}.'.format(
                        self.camera_port))
                    log('USB Camera not detected.',
                        message_type='error', title='take-photo')

    def save(self, filename_only=False, add_timestamp=True):
        """Save captured image."""
        if self.directory is None:
            directory = os.path.dirname(os.path.realpath(__file__)) + os.sep
            try:
                testfilename = directory + 'test_write.try_to_write'
                testfile = open(testfilename, 'w')
                testfile.close()
                os.remove(testfilename)
            except IOError:
                directory = '/tmp/images/'
        else:
            directory = self.directory
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
        if 'NONE' in CAMERA:
            log('No camera selected. Choose a camera on the device page.',
                message_type='error', title='take-photo')
            sys.exit(0)
        elif 'RPI' in CAMERA:
            # With Raspberry Pi Camera:
            image_filename = self.save(filename_only=True)
            try:
                retcode = call(['raspistill', '-w', '640', '-h', '480',
                                '-o', image_filename])
            except OSError:
                log('Raspberry Pi Camera not detected.',
                    message_type='error', title='take-photo')
                sys.exit(0)
            else:
                if retcode == 0:
                    print('Image saved: {}'.format(image_filename))
                    return image_filename
                else:
                    log('Problem getting image.',
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
                log('Problem getting image.',
                    message_type='error', title='take-photo')
                sys.exit(0)
            self.image_captured = True
            return self.save()


if __name__ == '__main__':
    Capture().capture()
