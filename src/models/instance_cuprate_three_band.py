#!/usr/bin/env python3

from .model import Model
from crystals import Tetra, Hexa
from ebands import *
from numpy import cos, sin, exp

cuprate_three_band = Model(Eband_cuprate_three_band,
                        Tetra(),
                        'cuprate_three_band',
                        3,
                        Ematrix_cuprate_three_band)
