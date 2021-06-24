# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


setup(
    name='timesat_io',
    version='0.0.1',
    description='A package for interacting with TIMESAT data.',
    author='Adam Weingram',
    author_email='aweingram@ucmerced.edu',
    url='https://github.com/adamweingram/timesat_io',
    packages=find_packages(exclude=('tests', 'docs'))
)
