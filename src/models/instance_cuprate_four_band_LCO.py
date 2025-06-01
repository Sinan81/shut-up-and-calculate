#!/usr/bin/env python3

from .model import Model
from crystals import Tetra, Hexa
from ebands import *
from numpy import cos, sin, exp

cuprate_four_band_LCO = Model(Eband_LCO_four_band,
                            Tetra(),
                            'cuprate_four_band_LCO',
                            4,
                            Ematrix_LCO_four_band)
