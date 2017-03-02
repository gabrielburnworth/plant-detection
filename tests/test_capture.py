#!/usr/bin/env python
"""Capture Tests

For Plant Detection.
"""
import unittest
import fakeredis
from PD.Capture import Capture


class GetCoordinatesTest(unittest.TestCase):
    """Check coordinate retrieval from redis"""

    def setUp(self):
        self.coordinates = [300, 500]
        r = fakeredis.FakeStrictRedis()
        r.lpush('BOT_STATUS.location', self.coordinates[1])
        r.lpush('BOT_STATUS.location', self.coordinates[0])
        self.capture = Capture(r=r)
        self.capture.silent = True

    def test_get_coordinates(self):
        """Get location from redis"""
        self.assertEqual(self.capture.getcoordinates(), self.coordinates)
