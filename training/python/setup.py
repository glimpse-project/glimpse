#! /usr/bin/env python

from setuptools import setup, Extension
from distutils.command.build import build

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

class BuildExtFirst(build):
    sub_commands = [('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_clib', build.has_c_libraries),
                    ('build_scripts', build.has_scripts)]

glimpse = Extension('_glimpse',
        ['glimpse.i', 'glimpse_python.cc'],
        include_dirs = [numpy_include],
        libraries = ['glimpse'])

setup(name = 'glimpse',
      description = 'A library for performing joint inference from depth images',
      author = 'Robert Bragg, Chris Lord',
      author_email = 'robert@impossible.com, chrisl@impossible.com',
      version = '0.0.1',
      package_dir = { 'glimpse': '.' },
      packages = [ 'glimpse' ],
      cmdclass = { 'build': BuildExtFirst },
      ext_modules = [ glimpse ],
      license = 'Proprietary',
      url = 'https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6',
      install_requires = [ 'numpy' ])
