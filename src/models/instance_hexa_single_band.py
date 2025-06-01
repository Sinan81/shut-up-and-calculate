#!/usr/bin/env python3

from .model import Model
from crystals import Tetra, Hexa
from ebands import *
from numpy import cos, sin, exp

hexa_single_band = Model(Eband_hexa, Hexa(), 'hexa_single_band')

cuprate_three_band = Model(Eband_cuprate_three_band, Tetra(), 'cuprate_three_band', 3, Ematrix_cuprate_three_band)
cuprate_four_band_LCO = Model(Eband_LCO_four_band, Tetra(), 'cuprate_four_band_LCO', 4, Ematrix_LCO_four_band)

tetra_single_band_ddw = Model(Eband_tetra_single_band_ddw,
                            Tetra(),
                            'tetra_single_band_ddw',
                            2,
                            Ematrix_tetra_single_band_ddw
                            )
tetra_single_band_ddw.isAFRBZ=True
