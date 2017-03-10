#!/usr/bin/env python
"""Capture Tests

For Plant Detection.
"""
import os
import unittest
import json
try:
    import fakeredis
    test_redis = True
except ImportError:
    test_redis = False
from PD import ENV


@unittest.skipUnless(test_redis, "requires fakeredis")
class LoadENVTest(unittest.TestCase):
    """Check data retrieval from redis"""

    def setUp(self):
        self.coordinates = [300, 500, -100]
        self.r = fakeredis.FakeStrictRedis()
        self.testvalue = u'some test data'
        self.testjson = {u"label": u"testdata", u"value": 5}
        self.badjson_string = '{"label": "whoop'

    def test_get_coordinates(self):
        """Get location from redis"""
        self.r.lpush('BOT_STATUS.location', self.coordinates[2])
        self.r.lpush('BOT_STATUS.location', self.coordinates[1])
        self.r.lpush('BOT_STATUS.location', self.coordinates[0])
        self.assertEqual(ENV.redis_load(
            'location', other_redis=self.r), self.coordinates)

    def test_no_coordinates(self):
        """Coordinates don't exist"""
        self.assertEqual(ENV.redis_load(
            'location', other_redis=self.r), None)

    def test_not_coordinates(self):
        """Coordinates aren't a list"""
        self.r.set('BOT_STATUS.location', 'notalist')
        self.assertEqual(ENV.redis_load(
            'location', other_redis=self.r), None)

    def test_env_load(self):
        """Get user_env from redis"""
        self.r.set('BOT_STATUS.user_env.testkey', self.testvalue)
        self.assertEqual(
            ENV.redis_load('user_env', name='testkey',
                           get_json=False, other_redis=self.r),
            self.testvalue)

    def test_json_env_load(self):
        """Get json user_env from redis"""
        self.r.set('BOT_STATUS.user_env.testdata', json.dumps(self.testjson))
        self.assertEqual(ENV.redis_load(
            'user_env', name='testdata', other_redis=self.r), self.testjson)

    def test_bad_json_env_load(self):
        """Try to get bad json user_env from redis"""
        self.r.set('BOT_STATUS.user_env.testdata', self.badjson_string)
        self.assertEqual(
            ENV.redis_load('user_env', name='testdata', other_redis=self.r),
            None)

    def test_none_user_env_load(self):
        """Try to get a non-existant user_env from redis"""
        self.assertEqual(
            ENV.redis_load('user_env', name='doesntexist', other_redis=self.r),
            None)

    def test_os_env_load(self):
        """Try to get an env from os"""
        os.environ['oktestenv'] = 'test'
        self.assertEqual(ENV.load_env('oktestenv', get_json=False), 'test')

    def test_none_os_env_load(self):
        """Try to get a non-existant env from os"""
        self.assertEqual(ENV.load_env('doesntexist'), None)

    def test_bad_json_os_env_load(self):
        """Try to get bad json env from os"""
        os.environ['testbadjson'] = '{"label": "whoop'
        self.assertEqual(ENV.load_env('testbadjson'), None)

    def tearDown(self):
        self.r.flushall()
