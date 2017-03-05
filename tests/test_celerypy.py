#!/usr/bin/env python
"""CeleryPy Tests

For Plant Detection.
"""
import unittest
import json
from PD import CeleryPy


class CeleryScript(unittest.TestCase):
    """Check celery script"""

    def setUp(self):
        self.add_point = CeleryPy.add_point(1, 2, 3, 4)
        self.add_point_static = {
            'kind': 'add_point', 'args': {'radius': 4, 'location': {
                'kind': 'coordinate', 'args': {'y': 2, 'x': 1, 'z': 3}}},
            'body': [{'kind': 'pair', 'args': {
                'value': 'plant-detection', 'label': 'created_by'}}]
        }
        self.set_env_var = CeleryPy.set_user_env('PLANT_DETECTION_options',
                                                 json.dumps({"in": "puts"}))
        self.set_env_var_static = {"kind": "set_user_env", "args": {},
                                   "body": [{"kind": "pair", "args": {
                                       "label": "PLANT_DETECTION_options",
                                       "value": "{\"in\": \"puts\"}"}}]}
        self.move_absolute_coordinate = CeleryPy.move_absolute(
            [10, 20, 30],
            [40, 50, 60],
            800)
        self.move_absolute_coordinate_static = {
            'kind': 'move_absolute', 'args': {'speed': 800, 'location': {
                'kind': 'coordinate', 'args': {'y': 20, 'x': 10, 'z': 30}},
                'offset': {'kind': 'coordinate',
                           'args': {'y': 50, 'x': 40, 'z': 60}}}
        }
        self.move_absolute_location = CeleryPy.move_absolute(
            ['tool', 1],
            [40, 50, 60],
            800)
        self.move_absolute_location_static = {
            'kind': 'move_absolute', 'args': {'speed': 800, 'location': {
                'kind': 'tool', 'args': {'tool_id': 1}},
                'offset': {'kind': 'coordinate',
                           'args': {'y': 50, 'x': 40, 'z': 60}}}
        }

    def test_add_point(self):
        """Check add_point celery script"""
        self.assertEqual(self.add_point_static, self.add_point)

    def test_set_env_var(self):
        """Check set_env_var celery script"""
        self.assertEqual(self.set_env_var_static, self.set_env_var)

    def test_move_absolute_coordinate(self):
        """Check move_absolute celery script with coordinates"""
        self.assertEqual(self.move_absolute_coordinate_static,
                         self.move_absolute_coordinate)

    def test_move_absolute_location(self):
        """Check move_absolute celery script with a location"""
        self.assertEqual(self.move_absolute_location_static,
                         self.move_absolute_location)
