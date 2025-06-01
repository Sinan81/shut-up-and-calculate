#!/usr/bin/env python3

from crystals import *

class Model:
    # Associate energy band & the crystal for convenience
    def __init__(self, Eband, crystal, name, rank=1, Ematrix=None):
        self.Eband = Eband
        self.Ematrix = Ematrix # only used in multiband calculations
        self.rank = rank # single or multi dim
        self.crystal = crystal
        self.__name__ = name
        self.jfactors = None # hand derived current sus factor
        self.hfactors_left = None # individual contribtions to a current sus factor along a bond
        self.hfactors_right = None
        self.gbasis   = None # used in calculating exchange interaction contributions
