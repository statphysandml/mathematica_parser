#!/usr/bin/env python

from distutils.core import setup

setup(name='mathematicaparser',
      version='0.1',
      description='Python modules for converting equations exported from mathematica into thrust code.',
      author='Lukas Kades',
      author_email='lukaskades@googlemail.com',
      url='https://github.com/statphysandml/mathematica_parser',
      packages=['mathematicaparser',
                'mathematicaparser.core',
                'mathematicaparser.odevisualization'],
     )
