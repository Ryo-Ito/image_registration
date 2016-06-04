"""
    This program is to use cython.
    1. Change the file's name in cythonize to a file you want to compile
    2. Type command: $ python setup.py build_ext -i
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("imageprocessing", ["imageprocessing.pyx"])]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()]
)