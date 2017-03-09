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


def create_node(kind=None, args=None):
    """Create a kind, args Celery Script node."""
    node = {}
    node['kind'] = kind
    node['args'] = args
    return node


def create_pair(label=None, value=None):
    """Create a label, value Celery Script node."""
    pair = {}
    pair['label'] = label
    pair['value'] = value
    return pair


def _saved_location_node(name, _id):
    args = {}
    args[name + '_id'] = _id
    saved_location = create_node(kind=name, args=args)
    return saved_location


def _coordinate_node(x_coord, y_coord, z_coord):
    coordinates = _encode_coordinates(x_coord, y_coord, z_coord)
    coordinate = create_node(kind='coordinate', args=coordinates)
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
    point = create_node(kind='add_point', args=args)
    created_by = create_pair(label='created_by', value='plant-detection')
    point['body'] = [create_node(kind='pair', args=created_by)]
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
    _set_user_env = create_node(kind='set_user_env', args={})
    env_var = create_pair(label=label, value=value)
    _set_user_env['body'] = [create_node(kind='pair', args=env_var)]
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
    _move_absolute = create_node(kind='move_absolute', args=args)
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
    _data_update = create_node(kind='data_update', args=args)
    if isinstance(ids_, list):
        body = []
        for id_ in ids_:
            _endpoint = create_pair(label=endpoint, value=str(id_))
            body.append(create_node(kind='pair', args=_endpoint))
    else:
        _endpoint = create_pair(label=endpoint, value=ids_)
        body = [create_node(kind='pair', args=_endpoint)]
    _data_update['body'] = body
    return _data_update
