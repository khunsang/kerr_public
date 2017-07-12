#!/usr/bin/env python

# Import useful things
from distutils.core import setup
from setuptools import find_packages

#
setup(name='kerr_public',
      version='1.0',
      description='Python Utilities for BH QNMs',
      author='Lionel London',
      author_email='lionel.london@ligo.org',
      packages=find_packages(),
      include_package_data=True,
      package_dir={'kerr': 'kerr'},
      url='https://github.com/llondon6/kerr_public/',
      download_url='https://github.com/llondon6/kerr_public/archive/master.zip',
      install_requires=['h5py','numpy','scipy','matplotlib'],
     )
