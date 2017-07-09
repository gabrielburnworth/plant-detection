#!/usr/bin/env python

"""Plant Detection package setup."""

from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

if __name__ == '__main__':
    setup(name='plant_detection',
          version='0.0.1',
          description='Detect and mark plants in a soil area image using Python OpenCV',
          long_description=readme,
          url='https://github.com/FarmBot-Labs/plant-detection',
          author='FarmBot Inc.',
          license='MIT',
          author_email='plantdetection@farmbot.io',
          packages=['plant_detection'],
          include_package_data=True,
          classifiers=[
              'Development Status :: 2 - Pre-Alpha',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.5',
              'Topic :: Scientific/Engineering :: Image Recognition',
          ],
          keywords=['farmbot', 'python', 'opencv'],
          #   install_requires=required,
          test_suite='plant_detection.tests.tests.test_suite',
          scripts=['quickscripts/load_and_detect.py'],
          zip_safe=False)
