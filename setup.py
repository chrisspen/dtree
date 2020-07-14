#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from distutils.core import setup, Command # pylint: disable=no-name-in-module

import dtree

class TestCommand(Command):
    description = "Runs unittests."
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('python dtree.py')

setup(
    name='dtree',
    version=dtree.__version__,
    description='A simple pure-Python batch decision tree construction algorithm.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/dtree',
    license='LGPL',
    py_modules=['dtree'],
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms=['OS Independent'],
#    test_suite='dtree',
    cmdclass={
        'test': TestCommand,
    },
)
