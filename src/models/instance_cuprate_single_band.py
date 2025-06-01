#!/usr/bin/env python3

from .model import Model
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
    return 4*sin(k[0] + k[1] + +q[0]/2 + q[1]/2)**2

# left hand side
def h1a(k,q):
    # k[0] is kx etc
    return exp(1j*(k[0]+q[0]))


def h1b(k,q):
    # k[0] is kx etc
    return exp(-1j*k[0])


def h2a(k,q):
    # k[0] is kx etc
    return exp(-1j*(k[0]+q[0]))


def h2b(k,q):
    # k[0] is kx etc
    return exp(1j*k[0])


def h3a(k,q):
    # k[1] is ky etc
    return exp(1j*(k[1]+q[1]))


def h3b(k,q):
    # k[1] is ky etc
    return exp(-1j*k[1])


def h4a(k,q):
    # k[1] is ky etc
    return exp(-1j*(k[1]+q[1]))


def h4b(k,q):
    # k[1] is ky etc
    return exp(1j*k[1])


# right hand side
def h1a_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h1a(kpq,mq)


def h1b_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h1b(kpq,mq)


def h2a_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h2a(kpq,mq)


def h2b_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h2b(kpq,mq)


def h3a_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h3a(kpq,mq)


def h3b_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h3b(kpq,mq)


def h4a_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h4a(kpq,mq)


def h4b_right(k,q):
    kpq = (k[0]+q[0], k[1]+q[1]) # k+q
    mq = (-q[0],-q[1])      # -q
    return h4b(kpq,mq)


# gbasis
def g1(k):
    return cos(k[0])

def g2(k):
    return sin(k[0])

def g3(k):
    return cos(k[1])

def g4(k):
    return sin(k[1])

def vmat_direct(qx,qy,U=0.5,V=0.5,Vnn=0.5):
    return U + V*( cos(qx) + cos(qy)) + Vnn*2*cos(qx)*cos(qy)

# current operator goes like J ~ c1*c2 - c2*c1, hence the pairs of h factors.
#hlist = [ (h1a, h1b), (h2a,h2b), (h3a,h3b), (h4a,h4b) ]
hlist = [ (h1a, h1b), (h2a,h2b)]
hlist_right = [ (h1a_right, h1b_right), (h2a_right, h2b_right) ]


# List of models
cuprate_single_band = Model(Eband_cuprate, Tetra(), 'cuprate_single_band')
# Note that these factors should be multiplied by t_ij**2
cuprate_single_band.jfactors = (jfact1, jfact2, jfact3)
cuprate_single_band.hfactors_left = hlist
cuprate_single_band.hfactors_right = hlist_right
cuprate_single_band.gbasis = [ g1, g2, g3, g4 , 1]
cuprate_single_band.vmat_direct = vmat_direct
cuprate_single_band.U = 0         # initialize local interaction
cuprate_single_band.V = 0        # initialize nearest neighbour interaction
cuprate_single_band.Vnn = 0     # initialize next nearest neighbour
cuprate_single_band.vbasis = None   # to be used in gRPA
