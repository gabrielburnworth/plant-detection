#!/usr/bin/env python
"""Plant Detection Test Suite

For Plant Detection.
"""
import unittest

if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover('.')
    unittest.TextTestRunner(verbosity=2).run(testsuite)
