#!/usr/bin/env python
"""DB Tests

For Plant Detection.
"""
import sys
import unittest
from PD.DB import DB


class DBTest(unittest.TestCase):
    """Check plant identification"""

    def setUp(self):
        self.outfile = open('db_text_output_test.txt', 'w')
        sys.stdout = self.outfile
        self.db = DB()
        self.db.plants['known'] = [{'x': 1000, 'y': 1000, 'radius': 100}]
        self.db.coordinate_locations = [[1000, 1000, 75],
                                        [1000, 825, 50],
                                        [800, 1000, 50],
                                        [1090, 1000, 75],
                                        [900, 900, 50],
                                        [1000, 1150, 50]
                                        ]
        self.remove = [{'radius': 50.0, 'x': 1000.0, 'y': 825.0},
                       {'radius': 50.0, 'x': 800.0, 'y': 1000.0}]
        self.safe_remove = [{'radius': 50.0, 'x': 900.0, 'y': 900.0},
                            {'radius': 50.0, 'x': 1000.0, 'y': 1150.0}]
        self.save = [{'radius': 75.0, 'x': 1000.0, 'y': 1000.0},
                     {'radius': 75.0, 'x': 1090.0, 'y': 1000.0},
                     ]
        self.db.identify()
        self.add_point = [{'body': [{'kind': 'pair', 'args': {
            'value': 'plant-detection', 'label': 'created_by'}}],
            'kind': 'add_point', 'args': {'radius': 50.0, 'location': {
                'kind': 'coordinate', 'args': {'y': 825.0, 'x': 1000.0, 'z': 0}}}},
            {'body': [{'kind': 'pair', 'args': {
                'value': 'plant-detection', 'label': 'created_by'}}],
             'kind': 'add_point', 'args': {'radius': 50.0, 'location': {
                 'kind': 'coordinate', 'args': {'y': 1000.0, 'x': 800.0, 'z': 0}}}}]

    def test_plant_id_remove(self):
        """Check plants to be removed"""
        self.assertEqual(self.remove, self.db.plants['remove'])

    def test_plant_id_save(self):
        """Check plants to be saved"""
        self.assertEqual(self.save, self.db.plants['save'])

    def test_plant_id_safe_remove(self):
        """Check plants to be safely removed"""
        self.assertEqual(self.safe_remove, self.db.plants['safe_remove'])

    def test_api_download(self):
        """Run (failing) plant download assuming no API_TOKEN ENV"""
        self.db.load_plants_from_web_app()
        self.assertEqual(self.db.errors, {'401': 1})

    def test_api_upload(self):
        """Run (failing) plant upload assuming no API_TOKEN ENV"""
        self.db.upload_weeds()
        self.assertEqual(self.db.errors, {'401': 1})

    def test_print_coordinates(self):
        """Print unidentified plant coordinate data"""
        self.db.print_coordinates()
        self.outfile.close()
        self.outfile = open('db_text_output_test.txt', 'r')
        self.assertEqual(sum(1 for line in self.outfile), 7)

    def test_cs_add_point(self):
        """Output Celery Script add_point"""
        add_point = self.db.output_celery_script()
        self.assertEqual(add_point, self.add_point)

    def test_save_to_tmp(self):
        """Save plants to file in tmp directory"""
        self.db.tmp_dir = "/tmp/"
        self.db.save_plants()

    def tearDown(self):
        self.outfile.close()
        sys.stdout = sys.__stdout__
