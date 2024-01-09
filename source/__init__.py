#!usr/bin/env python

__version__ = '0.0.0'

from functools import partial, wraps
import os
import numpy as np
import pandas as pd
import sklearn


# Absolute location of various directories relative to project installation.
PROJ_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
IMAGE_DIR = os.path.join(PROJ_DIR, 'images')
RUNTIME_DIR = os.path.join(PROJ_DIR, 'runtime')

# System wide settings.
settings = {
    'random_seed': 42,      # for reproducibility
}
