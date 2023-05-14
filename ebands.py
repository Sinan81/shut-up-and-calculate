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
from numpy import pi, sqrt
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import dblquad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from pathos.multiprocessing import ProcessingPool as PPool

matplotlib.use("TkAgg")

# set npool to number of cpus/threads the machine has
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2

ic = np.complex(0, 1.0)

kT = 0.01


@jit(nopython=True)
def Eband1_hexa(kx, ky):
    """
    make energy matrix
    """
    eps = 0
    t1 = 1
    a = 1
    theta_x = kx*a*0.5
    theta_y = ky*a*sqrt(3.)*0.5
    cx = np.cos(theta_x)
    sx = np.sin(theta_x)
    cy = np.cos(theta_y)
    sy = np.sin(theta_y)

    band = (
        eps
        - 2*t1*(cx**2 - sx**2)
        - 4*t1*cx*cy
    )
    return band



@jit(nopython=True)
def Eband1_cuprate(kx, ky):
    """
    make cuprate energy matrix:
    Tetra system with d-wave orbitals
    """
    eps = 0
    t1 = 1
    t2 = 0.0
    t3 = 0.0

    # 1-band cuprate
    band = (
        eps
        - 2 * t1 * (np.cos(kx) + np.cos(ky))
        - 4 * t2 * np.cos(kx) * np.cos(ky)
        - 2 * t3 * (np.cos(2 * kx) + np.cos(2 * ky))
    )
    return band


def Ematrix_LCO_four_band(kx,ky):
    """
    make energy matrix for LaCuO4
    """
    # Reference:
    # Unified description of cuprate superconductors using four-band d-p model
    # https://arxiv.org/abs/2105.11664
    t1 = 1.42
    t2 = 0.61
    t3 = 0.07
    t4 = 0.65
    t5 = 0.05
    t6 = 0.07
    eps_dx2y2 = -0.87
    eps_dz = -0.11
    eps_px = -3.13
    eps_py = -3.13

    ic = np.complex(0,1)
    t11 = eps_dx2y2
    t21 = 0
    t22 = eps_dz -2*t5*(np.cos(kx) + np.cos(ky) )
    t31 = 2*ic*t1*np.sin(kx/2)
    t32 = -2*ic*t4*np.sin(kx/2)
    t33 = eps_px + 2*t3*np.cos(kx) + 2*t6*( np.cos(kx+ky) + np.cos(kx - ky) )
    t41 = -2*ic*t1*np.sin(ky/2)
    t42 = -2*ic*t4*np.sin(ky/2)
    t43 = 2*t2*( np.cos( (kx+ky)/2 ) - np.cos( (kx-ky)/2 ) )
    t44 = eps_py + 2*t3*np.cos(ky) + 2*t6*(np.cos(kx + ky) + np.cos(kx-ky) )

    m = np.matrix([ [ t11, np.conj(t21),  np.conj(t31), np.conj(t41) ],
                  [ t21, t22,           np.conj(t32), np.conj(t42) ],
                  [ t31, t32,           t33,          np.conj(t43) ],
                  [ t41, t42,           t43,            t44        ]
                ])
    return m;

def Eband_LCO_four_band(kx,ky,iband=1,em=Ematrix_LCO_four_band):
    """
    make energy bands
    """
    vl,vc = np.linalg.eig(em(kx,ky))
    vl = np.sort(vl)
    return vl[iband]


def Ematrix_cuprate_three_band(kx,ky):
    """
    make energy matrix
    """
    ed = 0
    tpd = 1
    ctg = 2.5
    ex = ed-ctg
    ey = ex
    tpp = 0.5
    t = 1
    t1 = 1
    t2 = 0.#25
    t3 = 0.
    ic = np.complex(0,1)
    # 3-band case (Emery model for Cuprates)
    m = np.matrix([ [ ed, 2.*tpd*np.sin(kx/2.), -2.*tpd*np.sin(ky/2.) ],
                  [ 2.*tpd*np.sin(kx/2.), ex, -4.*tpp*np.sin(kx/2.)*np.sin(ky/2.)],
                  [ -2.*tpd*np.sin(ky/2.), -4.*tpp*np.sin(kx/2.)*np.sin(ky/2.), ey  ]
                ])
    return m;

def Eband_cuprate_three_band(kx,ky,iband=1,em=Ematrix_cuprate_three_band):
    """
    make energy bands
    """
    vl,vc = np.linalg.eig(em(kx,ky))
    vl = np.sort(vl)
    return vl[iband]

def make_Eall(X,Y,func_em):
    vl,vc = np.linalg.eig(func_em(0.,0.)) # get size
    Eall  = np.zeros((vl.size, X.size*Y.size));
    Evecs = np.zeros((vl.size**2, X.size*Y.size));
    # These naive for loops should be replaced with a more performant logic
    # While numpy vectorize worked nicely for getting eigenvalues (one band at a time!)
    # It caused complications in terms of getting eigenvectors as well.
    # TODO:
    # - Try numpy vectorize with function signature
    # - try parallellizing with multiprocess map
    # - try using numba
    i = 0
    for kx in X:
        for ky in Y:
            vl,vc = np.linalg.eig(func_em(kx,ky))
            Eall[:,i] = vl
            Evecs[:,i] = vc.flatten()
            i = i +1
    return Eall

def get_Evecs(X,Y,func_em):
    vl,vc = np.linalg.eig(func_em(0.,0.)) # get size
    Eall  = np.zeros((vl.size, X.size*Y.size));
    Evecs = np.zeros((vl.size**2, X.size*Y.size));
    # These naive for loops should be replaced with a more performant logic
    # While numpy vectorize worked nicely for getting eigenvalues (one band at a time!)
    # It caused complications in terms of getting eigenvectors as well.
    # TODO:
    # - Try numpy vectorize with function signature
    # - try parallellizing with multiprocess map
    # - try using numba
    i = 0
    for kx in X:
        for ky in Y:
            vl,vc = np.linalg.eig(func_em(kx,ky))
            Eall[:,i] = vl
            Evecs[:,i] = vc.flatten()
            i = i +1
    return Evecs


def eband(kx,ky,iband,em):
    """
    make energy bands
    """
    vl,vc = np.linalg.eig(em(kx,ky))
    vl = np.sort(vl)
    return vl[iband]

