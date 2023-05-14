#!/usr/bin/env python3

from crystals import Tetra, Hexa
from ebands import *

class Model:
    # Associate energy band & the crystal for convenience
    def __init__(self, Eband, crystal, name, rank=1, Ematrix=None):
        self.Eband = Eband
        self.Ematrix = Ematrix # only used in multiband calculations
        self.rank = rank # single or multi dim
        self.crystal = crystal
        self.__name__ = name

# List of models
cuprate_single_band = Model(Eband1_cuprate, Tetra(), 'cuprate_single_band')
hexa_single_band = Model(Eband1_hexa, Hexa(), 'hexa_single_band')
cuprate_three_band = Model(Eband_cuprate_three_band, Tetra(), 'cuprate_three_band', 3, Ematrix_cuprate_three_band)
cuprate_four_band_LCO = Model(Eband_LCO_four_band, Tetra(), 'cuprate_four_band_LCO', 4, Ematrix_LCO_four_band)

list_of_models = [  cuprate_single_band.__name__,
                    hexa_single_band.__name__,
                    cuprate_three_band.__name__]
