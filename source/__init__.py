#!usr/bin/env python

__version__ = '0.0.0'

import os

# Absolute location of various directories relative to project installation.
PROJ_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
IMAGE_DIR = os.path.join(PROJ_DIR, 'images')

# System wide settings.
settings = {
    'random_seed': 42,      # for reproducibility
}
