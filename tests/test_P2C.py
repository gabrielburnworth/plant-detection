#!/usr/bin/env python
"""P2C Tests

For Plant Detection.
"""
import sys
import cv2
import unittest
from PD.P2C import Pixel2coord
from PD.DB import DB


class P2CTest(unittest.TestCase):
    """Check calibration"""

    def setUp(self):
        self.outfile = open('p2c_text_output_test.txt', 'w')
        sys.stdout = self.outfile
        self.db = DB()
        self.two_objects = cv2.imread('PD/p2c_test_calibration.jpg', 1)
        self.three_objects = self.two_objects.copy()
        self.one_object = self.two_objects.copy()
        cv2.circle(self.one_object,
                   (175, 475), int(50),
                   (255, 255, 255), -1)
        cv2.circle(self.three_objects,
                   (600, 300), int(25),
                   (0, 0, 255), -1)
        cv2.imwrite('image.jpg', self.three_objects)

    def test_one_object(self):
        """Detect one object during calibration"""
        db = DB()
        p2c = Pixel2coord(db, calibration_image=self.one_object)
        exit_flag = p2c.calibration()
        self.assertEqual(db.object_count, 1)

    def test_two_objects(self):
        """Detect two objects during calibration"""
        db = DB()
        p2c = Pixel2coord(db, calibration_image=self.two_objects)
        exit_flag = p2c.calibration()
        self.assertEqual(db.object_count, 2)

    def test_three_objects(self):
        """Detect three objects during calibration"""
        db = DB()
        p2c = Pixel2coord(db, calibration_image=self.three_objects)
        exit_flag = p2c.calibration()
        self.assertEqual(db.object_count, 3)

    def tearDown(self):
        self.outfile.close()
        sys.stdout = sys.__stdout__
