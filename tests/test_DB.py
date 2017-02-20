#!/usr/bin/env python
"""DB Tests

For Plant Detection.
"""
import unittest
from PD.DB import DB


class DBTest(unittest.TestCase):
    """Check plant identification"""

    def setUp(self):
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

    def test_plant_id_remove(self):
        """Check plants to be removed"""
        self.db.identify()
        self.assertEqual(self.remove, self.db.plants['remove'])

    def test_plant_id_save(self):
        """Check plants to be saved"""
        self.db.identify()
        self.assertEqual(self.save, self.db.plants['save'])

    def tearDown(self):
        self.db.plants['known'] = None
