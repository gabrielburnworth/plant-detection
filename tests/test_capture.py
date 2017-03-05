#!/usr/bin/env python
"""Capture Tests

For Plant Detection.
"""
import os
import sys
import unittest
try:
    import fakeredis
    test_redis = True
except ImportError:
    test_redis = False
from PD.Capture import Capture


class GetCoordinatesTest(unittest.TestCase):
    """Check coordinate retrieval from redis"""

    def setUp(self):
        self.coordinates = [300, 500, -100]
        r = fakeredis.FakeStrictRedis()
        r.lpush('BOT_STATUS.location', self.coordinates[2])
        r.lpush('BOT_STATUS.location', self.coordinates[1])
        r.lpush('BOT_STATUS.location', self.coordinates[0])
        self.capture = Capture(r=r)
        self.capture.silent = True

    @unittest.skipUnless(test_redis, "requires fakeredis")
    def test_get_coordinates(self):
        """Get location from redis"""
        self.assertEqual(self.capture.getcoordinates(), self.coordinates)


class CheckCameraTest(unittest.TestCase):
    """Check for camera"""

    def setUp(self):
        self.nullfile = open(os.devnull, 'w')
        sys.stdout = self.nullfile

    def test_camera_check(self):
        """Test camera check"""
        Capture().camera_check()

    def tearDown(self):
        self.nullfile.close()
        sys.stdout = sys.__stdout__
