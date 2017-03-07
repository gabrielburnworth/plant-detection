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
        """Send Celery Script or return the JSON string.

        Celery Script is sent by prefixing the string in the `BEGIN_CS`
        environment variable.
        """
        try:
            begin_cs = os.environ['BEGIN_CS']
        except KeyError:
            return function(*args, **kwargs)
        else:
            print(begin_cs + json.dumps(function(*args, **kwargs)))
            return
    return wrapper


def _encode_coordinates(x_coord, y_coord, z_coord):
    coords = {}
    coords['x'] = x_coord
    coords['y'] = y_coord
    coords['z'] = z_coord
    return coords


def _create_node(kind, args):
    node = {}
    node['kind'] = kind
    node['args'] = args
    return node


def _create_pair(label, value):
    pair = {}
    pair['label'] = label
    pair['value'] = value
    return pair


def _saved_location_node(name, _id):
    args = {}
    args[name + '_id'] = _id
    saved_location = _create_node(name, args)
    return saved_location


def _coordinate_node(x_coord, y_coord, z_coord):
    coordinates = _encode_coordinates(x_coord, y_coord, z_coord)
    coordinate = _create_node('coordinate', coordinates)
    return coordinate


@_print_json
def add_point(point_x, point_y, point_z, point_r):
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
    args['location'] = _coordinate_node(point_x, point_y, point_z)
    args['radius'] = point_r
    point = _create_node('add_point', args)
    created_by = _create_pair('created_by', 'plant-detection')
    point['body'] = [_create_node('pair', created_by)]
    return point


@_print_json
def set_user_env(label, value):
    """Celery Script to set an environment variable.

    Kind:
        set_user_env
    Body:
        Kind: pair
        Args:
            label: <ENV VAR name>
            value: <ENV VAR value>
    """
    _set_user_env = _create_node('set_user_env', {})
    env_var = _create_pair(label, value)
    _set_user_env['body'] = [_create_node('pair', env_var)]
    return _set_user_env


@_print_json
def move_absolute(location, offset, speed):
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
        args['location'] = _saved_location_node(
            location[0], location[1])
    if len(location) == 3:
        args['location'] = _coordinate_node(*location)
    args['offset'] = _coordinate_node(*offset)
    args['speed'] = speed
    _move_absolute = _create_node('move_absolute', args)
    return _move_absolute


@_print_json
def data_update(endpoint, ids_):
    """Celery Script to signal that a sync is required.

    Kind:
        data_update
    Args:
        value: updated
    Body:
        Kind: pair
        Args:
            label: endpoint
            value: id
    """
    args = {}
    args['value'] = 'updated'
    _data_update = _create_node('data_update', args)
    if isinstance(ids_, list):
        body = []
        for id_ in ids_:
            _endpoint = _create_pair(endpoint, str(id_))
            body.append(_create_node('pair', _endpoint))
    else:
        _endpoint = _create_pair(endpoint, ids_)
        body = [_create_node('pair', _endpoint)]
    _data_update['body'] = body
    return _data_update
