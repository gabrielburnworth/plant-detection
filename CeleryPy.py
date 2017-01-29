#!/usr/bin/env python
"""Celery Py.

Python wrappers for FarmBot Celery Script JSON nodes.
"""
import json
from functools import wraps

class FarmBotJSON():
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

    def _saved_location_node(self, name, id):
        args = {}
        args[name + '_id'] = id
        saved_location = self._create_node(name, args)
        return saved_location

    def _coordinate_node(self, x, y, z):
        coordinates = self._encode_coordinates(x, y, z)
        coordinate = self._create_node('coordinate', coordinates)
        return coordinate

    def _print_json(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            print json.dumps(function(*args, **kwargs),
                             indent=2, separators=(',', ': '))
        return wrapper

    @_print_json
    def add_point(self, x, y, z):
        """Kind:
                point
           Arguments:
                Location:
                    Coordinate (x, y, z)
        """
        args = {}
        args['location'] = self._coordinate_node(x, y, z)
        point = self._create_node('point', args)
        return point

    @_print_json
    def move_absolute(self, location, offset, speed):
        """Kind:
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
            args['location'] = self._saved_location_node(location[0], location[1])
        if len(location) == 3:
            args['location'] = self._coordinate_node(*location)
        args['offset'] = self._coordinate_node(*offset)
        args['speed'] = speed
        move_absolute = self._create_node('move_absolute', args)
        return move_absolute

    @_print_json
    def add_plant(self, plant_id, location, radius):
        """Kind:
                add_plant
           Arguments:
                plant_id
                Radius
                Location:
                    Coordinate (x, y, z)
        """
        args = {}
        if isinstance(location, int):
            args['location'] = self._saved_location_node('plant', plant_id)
        if len(location) == 3:
            args['location'] = self._coordinate_node(*location)
        args['plant_id'] = plant_id
        args['radius'] = radius
        move_absolute = self._create_node('plant', args)
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

    FarmBot = FarmBotJSON()
    FarmBot.add_point(x, y, z)
    print
    FarmBot.move_absolute([x, y, z], [x_offset, y_offset, z_offset], speed)
    print
    FarmBot.move_absolute(tool, [x_offset, y_offset, z_offset], speed)
    print
    FarmBot.move_absolute(plant, [x_offset, y_offset, z_offset], speed)
    print
    FarmBot.add_plant(plant_id, [x, y, z], radius)
