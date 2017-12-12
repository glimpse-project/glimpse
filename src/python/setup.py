#! /usr/bin/env python
#
# Copyright (c) 2017 Glimp IP Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

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
        ['glimpse.i', 'glimpse_python.cc',
         '../image_utils.cc',
         '../infer.cc',
         '../loader.cc',
         '../tinyexr.cc',
         '../parson.c',
         '../llist.c',
         '../xalloc.c'
        ],
        include_dirs = [numpy_include],
        libraries = ['png'])

setup(name = 'glimpse',
      description = 'A library for performing joint inference from depth images',
      author = 'Robert Bragg, Chris Lord',
      author_email = 'robert@impossible.com, chrisl@impossible.com',
      version = '0.0.1',
      package_dir = { 'glimpse': '.' },
      packages = [ 'glimpse' ],
      cmdclass = { 'build': BuildExtFirst },
      ext_modules = [ glimpse ],
      license = 'MIT',
      url = 'https://medium.com/impossible/glimpse-a-sneak-peak-into-your-creative-self-29bd2e656ff6',
      install_requires = [ 'numpy' ])
