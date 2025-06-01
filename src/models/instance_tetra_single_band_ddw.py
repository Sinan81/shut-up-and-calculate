#!/usr/bin/env python3

from .model import Model
from crystals import Tetra, Hexa
from ebands import *
from numpy import cos, sin, exp

tetra_single_band_ddw = Model(Eband_tetra_single_band_ddw,
                            Tetra(),
                            'tetra_single_band_ddw',
                            2,
                            Ematrix_tetra_single_band_ddw
                            )

tetra_single_band_ddw.isAFRBZ=True
