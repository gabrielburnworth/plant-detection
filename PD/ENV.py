#!/usr/bin/env python
"""ENV.

Load and save environment variables.
"""
import os
import json
try:
    import redis
except ImportError:
    REDIS = False
else:
    REDIS = True
from PD import CeleryPy


def _load_json(string):
    try:
        value = json.loads(string)
    except (TypeError, ValueError):
        value = None
    return value


def load_env(name, get_json=True):
    """Load an environment variable from OS."""
    try:
        env = os.environ[name]
    except KeyError:
        return

    if get_json and env is not None:
        value = _load_json(env)
    else:
        value = env
    return value


def redis_load(key, name=None, get_json=True, other_redis=None):
    """Load a value from redis."""
    value = None
    if REDIS:
        if other_redis is not None:
            _redis = other_redis
        else:
            _redis = redis.StrictRedis()
        try:
            _redis.ping()
        except redis.exceptions.ConnectionError:
            return

        if key == 'location':
            temp = _redis.lrange('BOT_STATUS.location', 0, -1)
            if temp and isinstance(temp, list):
                value = [int(coordinate) for coordinate in temp]
        else:
            temp = _redis.get('BOT_STATUS.{}.{}'.format(key, name))
            if temp is None:
                return

            decoded = temp.decode('utf-8')
            if get_json:
                value = _load_json(decoded)
            else:
                value = decoded
    return value


def load(name, get_json=True):
    """Load an environment variable (prioritize redis)."""
    value = redis_load('user_env', name=name, get_json=get_json)
    if value is None:
        value = load_env(name, get_json=get_json)
    return value


def save(name, value, its_json=True):
    """Save an environment variable to env and, if available, redis."""
    if its_json:
        value = json.dumps(value)
    unsent_cs = CeleryPy.set_user_env(name, value)
    os.environ[name] = value
    return unsent_cs
