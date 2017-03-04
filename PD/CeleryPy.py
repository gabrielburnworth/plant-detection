#!/usr/bin/env python
"""Celery Py.

Python wrappers for FarmBot Celery Script JSON nodes.
"""
import os
import json
from functools import wraps


def _print_json(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            print('{} {}'.format(os.environ['BEGIN_CS'],
                                 json.dumps(function(*args, **kwargs))))
        except KeyError:
            return function(*args, **kwargs)
    return wrapper


class CeleryPy(object):
    """Python wrappers for FarmBot Celery Script."""

    def _encode_coordinates(self, x, y, z):
        coords = {}
        coords['x'] = x
        coords['y'] = y
        coords['z'] = z
        return coords

    def _create_node(self, kind, args):
        node = {}
        node['kind'] = kind
        node['args'] = args
        return node

    def _create_pair(self, label, value):
        pair = {}
        pair['label'] = label
        pair['value'] = value
        return pair

    def _saved_location_node(self, name, _id):
        args = {}
        args[name + '_id'] = _id
        saved_location = self._create_node(name, args)
        return saved_location

    def _coordinate_node(self, x, y, z):
        coordinates = self._encode_coordinates(x, y, z)
        coordinate = self._create_node('coordinate', coordinates)
        return coordinate

    @_print_json
    def add_point(self, x, y, z, r):
        """Celery Script to add a point to the database.

        Kind:
            add_point
        Arguments:
            Location:
                Coordinate (x, y, z)
            Radius: r
        Body:
            Kind: pair
            Args:
                label: created_by
                value: plant-detection
        """
        args = {}
        args['location'] = self._coordinate_node(x, y, z)
        args['radius'] = r
        point = self._create_node('add_point', args)
        created_by = self._create_pair('created_by', 'plant-detection')
        point['body'] = [self._create_node('pair', created_by)]
        return point

    @_print_json
    def set_user_env(self, label, value):
        """Celery Script to set an environment variable.

        Kind:
            set_user_env
        Body:
            Kind: pair
            Args:
                label: <ENV VAR name>
                value: <ENV VAR value>
        """
        set_user_env = self._create_node('set_user_env', {})
        env_var = self._create_pair(label, value)
        set_user_env['body'] = [self._create_node('pair', env_var)]
        return set_user_env

    @_print_json
    def move_absolute(self, location, offset, speed):
        """Celery Script to move to a location.

        Kind:
            move_absolute
        Arguments:
            Location:
                Coordinate (x, y, z) or Saved Location ['tool', tool_id]
            Offset:
                Distance (x, y, z)
            Speed:
                Speed (mm/s)
        """
        args = {}
        if len(location) == 2:
            args['location'] = self._saved_location_node(
                location[0], location[1])
        if len(location) == 3:
            args['location'] = self._coordinate_node(*location)
        args['offset'] = self._coordinate_node(*offset)
        args['speed'] = speed
        move_absolute = self._create_node('move_absolute', args)
        return move_absolute

if __name__ == "__main__":
    x, y, z = 75.40, 30.00, -100.03
    x_offset, y_offset, z_offset = 10.00, 20.00, 10.00
    speed = 800
    tool_id = 54
    tool = ['tool', tool_id]
    plant_id = 10
    plant = ['plant', plant_id]
    radius = 48.00

    FarmBot = CeleryPy()
    print(FarmBot.add_point(x, y, z, radius))
    print
    print(FarmBot.move_absolute(
        [x, y, z],
        [x_offset, y_offset, z_offset],
        speed))
    print
    print(FarmBot.move_absolute(
        tool,
        [x_offset, y_offset, z_offset],
        speed))
    print
    print(FarmBot.move_absolute(
        plant,
        [x_offset, y_offset, z_offset],
        speed))
