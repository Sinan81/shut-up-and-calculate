#!/usr/bin/env python3

from numpy import pi, sqrt
import numpy as np

class Hexa:
    """
    hexa crystal specific stuff
    """
    def __init__(self):
        self.integ_xmin = -2*pi
        self.integ_xmax =  2*pi
        # slope and intersection of certain hexagon edges
        # for a hexagon with two sides perpendicular to x-axis
        isect = 4*pi/sqrt(3) # axis intersection
        ms = np.tan(pi/6.) # slope tan(30 degrees)
        # ymin
        self.gfun = lambda kx:  ms*kx - isect if kx > 0 else -(ms*kx + isect)
        # ymax
        self.hfun = lambda kx: -ms*kx + isect if kx > 0 else ms*kx + isect

        # area of first brilloin zone
        sq3 = sqrt(3.)
        self.fbz_area = 4*(pi**2)*3*sqrt(3)/2.

        # rectangular primitive cell for easy integration
        self.pc_kx_min = -2*pi
        self.pc_kx_max =  2*pi
        self.pc_ky_min = -2*pi/sqrt(3)
        self.pc_ky_max =  4*pi/sqrt(3)


    @staticmethod
    def overlay_FBZ(plt):
        """
        Draw a hexagon. Rotated 30 degrees with respect to
        real space Weigner-Seitz cell
        """
        s30 = np.sin(pi/6.)
        s60 = np.sin(pi/3.)
        c30 = np.cos(pi/6.)
        c60 = np.cos(pi/3.)

        # radious of imaginary circle enclosing the polygon
        radi = pi/(c30**2)
        hx = radi*np.array([1.,  c60, -c60,  -1., -c60,  c60, 1.])
        hy = radi*np.array([0.,  s60,  s60,   0., -s60, -s60, 0.])

        plt.plot(hx/pi,hy/pi, label='FBZ')
        plt.legend(loc='best')


    def get_kpoints(self,Nk=None,dk=None,isAFRBZ=False):
        # Nk: number of k points along 1 dimension
        # dk: k point grid size
        if not dk:
            dk = 2*pi/Nk

        if isAFRBZ: # Reduced Brilloin Zone for AF and like wavevector Q=(pi,pi)
            pass # TODO

        lkx = np.arange(self.pc_kx_min, self.pc_kx_max, dk)
        lky = np.arange(self.pc_ky_min, self.pc_ky_max, dk)
        xx,yy = np.meshgrid(lkx,lky)
        return xx,yy


class Tetra:
    """
    tetra crysctal specific stuff
    """

    G = (0,0)
    X = (pi,0)
    M = (pi,pi)
    sym_cuts = ( (G,X), (X,M), (M,G))


    def __init__(self):
        self.integ_xmin = 0.0
        self.integ_xmax = 2.0 * pi
        self.gfun = lambda ky: 0.0
        self.hfun = lambda ky: 2.0 * pi

        # area of first brilloin zone
        self.fbz_area = 4*(pi**2)

        # primitive cell for easy integration
        self.pc_kx_min = -pi
        self.pc_kx_max =  pi
        self.pc_ky_min = -pi
        self.pc_ky_max =  pi


    @staticmethod
    def overlay_FBZ(plt):
        'Draw a tetra FBZ'
        hx = np.array([pi, -pi, -pi, pi, pi])
        hy = np.array([pi,  pi, -pi,-pi, pi])
        plt.plot(hx/pi,hy/pi, label='FBZ')
        plt.legend(loc='best')


    def get_kpoints(self,Nk=None,dk=None,isAFRBZ=False):
        # Nk: number of k points along 1 dimension
        # dk: k point grid size
        if not dk:
            dk = 2*pi/Nk

        if isAFRBZ: # Reduced Brilloin Zone for AF and like wavevector Q=(pi,pi)
            pass # TODO

        lkx = np.arange(self.pc_kx_min, self.pc_kx_max, dk)
        lky = np.arange(self.pc_ky_min, self.pc_ky_max, dk)
        xx,yy = np.meshgrid(lkx,lky)
        return xx,yy

