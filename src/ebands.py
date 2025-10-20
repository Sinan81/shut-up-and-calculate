#!/usr/bin/env python3

# Tight binding approximation of solid state systems

# builtin modules
import time
from multiprocessing import Pool
import os
import pickle
import warnings

# extras
# numba results in 30x speed up!!!
from numba import jit
import numba
from numpy import pi, sqrt, cos, sin
import numpy as np


def make_Eall(xx,yy,func_em):
    # xx, yy are meshgrids for kx, ky
    # These naive for loops should be replaced with a more performant logic
    # While numpy vectorize worked nicely for getting eigenvalues (one band at a time!)
    # It caused complications in terms of getting eigenvectors as well.
    # TODO:
    # - Try numpy vectorize with function signature
    # - try parallellizing with multiprocess map
    # - try using numba
    i = 0
    points = list(zip(xx.ravel(), yy.ravel()))
    vl,vc = np.linalg.eig(func_em(0.,0.)) # get size
    Eall  = np.zeros((vl.size, len(points)));
    Evecs = np.zeros((vl.size**2, len(points)));
    for point in points:
        kx,ky = point
        vl,vc = np.linalg.eig(func_em(kx,ky))
        Eall[:,i] = vl
        Evecs[:,i] = vc.flatten()
        i = i +1
    return Eall

def get_Evecs(xx,yy,func_em, flatten=True):
    # xx, yy are meshgrids for kx, ky
    # These naive for loops should be replaced with a more performant logic
    # While numpy vectorize worked nicely for getting eigenvalues (one band at a time!)
    # It caused complications in terms of getting eigenvectors as well.
    # TODO:
    # - Try numpy vectorize with function signature
    # - try parallellizing with multiprocess map
    # - try using numba
    i = 0
    points = list(zip(xx.ravel(), yy.ravel()))
    vl,vc = np.linalg.eig(func_em(0.,0.)) # get size
    #Eall_flat = np.zeros((vl.size, len(points)));
    Evecs_flat = np.zeros((vl.size**2, len(points)));

    Eall = np.zeros((len(points),vl.size));
    Evecs = np.zeros((len(points),*vc.shape));
    for point in points:
        kx,ky = point
        vl,vc = np.linalg.eig(func_em(kx,ky))
        #Eall_flat[:,i] = vl
        Evecs_flat[:,i] = vc.flatten()
        Eall[i,:] = vl
        Evecs[i,:] = vc
        i = i +1
    if flatten:
        return Evecs_flat
    else:
        return Evecs, Eall
