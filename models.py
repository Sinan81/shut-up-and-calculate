#!/usr/bin/env python3

from crystals import Tetra, Hexa
from ebands import *
from numpy import cos, sin, exp

# dont convert these to class methods
# since numba will probably not work
def jfact1(k,q):
    return 4*sin(k[0] + q[0]/2)**2


def jfact2(k,q):
    return 4*sin(k[1] + q[1]/2)**2


def jfact3(k,q):
    kx=k[0]
    qx=q[0]
    ky=k[1]
    qy=q[1]
    AA = exp(1j*(kx + ky + qx))
    AB = -exp(1j*(kx - ky + qx - qy))
    BA = -exp(1j*(-kx+ky))
    BB = exp(-1j*(kx+ky+qy))
    return AA + AB + BA + BB


def vmat_direct(qx,qy,U=0.5,V=0.5,Vnn=0.5):
    return U + V*( cos(qx) + cos(qy)) + Vnn*2*cos(qx)*cos(qy)

class Model:
    # Associate energy band & the crystal for convenience
    def __init__(self, Eband, crystal, name, rank=1, Ematrix=None):
        self.Eband = Eband
        self.Ematrix = Ematrix # only used in multiband calculations
        self.rank = rank # single or multi dim
        self.crystal = crystal
        self.__name__ = name
        self.jfactors = None

# List of models
cuprate_single_band = Model(Eband1_cuprate, Tetra(), 'cuprate_single_band')
# Note that these factors should be multiplied by t_ij**2
cuprate_single_band.jfactors = (jfact1, jfact2)
cuprate_single_band.vmat_direct = vmat_direct
cuprate_single_band.U = 0         # initialize local interaction
cuprate_single_band.V = 0        # initialize nearest neighbour interaction
cuprate_single_band.Vnn = 0     # initialize next nearest neighbour
cuprate_single_band.vbasis = None   # to be used in gRPA

hexa_single_band = Model(Eband1_hexa, Hexa(), 'hexa_single_band')
cuprate_three_band = Model(Eband_cuprate_three_band, Tetra(), 'cuprate_three_band', 3, Ematrix_cuprate_three_band)
cuprate_four_band_LCO = Model(Eband_LCO_four_band, Tetra(), 'cuprate_four_band_LCO', 4, Ematrix_LCO_four_band)

def get_list_of_models():
    print('List of all models:')
    print('====================')
    for x in globals():
        if type(eval(x)) == Model:
            print(x)
