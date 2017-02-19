#!/usr/bin/env python
"""CeleryPy Tests

For Plant Detection.
"""
import unittest
import json
from PD.CeleryPy import CeleryPy


class CeleryScript(unittest.TestCase):
    """Check celery script"""

    def setUp(self):
        self.cs = CeleryPy()
        self.add_point = self.cs.add_point(1, 2, 3, 4)
        self.add_point_static = {
            'kind': 'add_point', 'args': {'radius': 4, 'location': {
                'kind': 'coordinate', 'args': {'y': 2, 'x': 1, 'z': 3}}},
            'body': [{'kind': 'pair', 'args': {
                'value': 'plant-detection', 'label': 'created_by'}}]
        }
        self.set_env_var = self.cs.set_user_env('PLANT_DETECTION_options',
                                                json.dumps({"in": "puts"}))
        self.set_env_var_static = {"kind": "set_user_env", "args": {},
                                   "body": [{"kind": "pair", "args": {
                                       "label": "PLANT_DETECTION_options",
                                       "value": "{\"in\": \"puts\"}"}}]}

    def test_add_point(self):
        """Check add_point celery script"""
        self.assertEqual(self.add_point_static, self.add_point)

    def test_set_env_var(self):
        """Check set_env_var celery script"""
        self.assertEqual(self.set_env_var_static, self.set_env_var)
