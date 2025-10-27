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
from numba import jit, njit, float64
import numba
from numpy import pi, sqrt, exp
import numpy as np
import scipy.integrate as integrate
from scipy.integrate import dblquad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from pathos.multiprocessing import ProcessingPool as PPool
#import pysnooper
import pdb

import warnings
warnings.filterwarnings('ignore')

#from models import *
from chi import *
from ebands import *
from util import *
from spectra import *
from crystals import *

if os.environ.get('MPLBACKEND') is None:
    matplotlib.use("TkAgg")

# set npool to number of cpus/threads the machine has
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2

ic = complex(0, 1.0)

kT = 0.01


class System:
    def __init__(self, filling=None):
        pass
#        self.model = model
#        self.crystal = model.crystal
#        self.Eband = model.Eband
#        self.filling = self.set_filling(filling)
#        self.eFermi = self.get_Fermi_level1(self.filling)
#        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
#        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
#        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
#        self.__name__ = model.__name__
#        self.spectra = Spectra(self)

    def get_default_eband(self):
        if self.crystal is Tetra:
            return Eband_cuprate

    def set_filling(self, filling):
        if not filling:
            if hasattr(self,'isAFRBZ'):
                # account for double counting in AF RBZ system
                return self.rank/2 - 0.5
            else: # normal system
                return self.rank - 0.5
        else:
            return filling

    def make_Eall1(self, xx, yy):
        veband = np.vectorize(self.Eband)
        # xx,yy are meshgrids
        Eall = veband(xx,yy)
        return Eall

    def get_Eall(self, X, Y):
        if self.rank == 1:
            Eall = self.make_Eall1(X,Y)
        else:
            Eall = make_Eall(X,Y, self.Ematrix)
            Eall = np.sort(Eall)
        return Eall

    def filling_vs_energy(self, isSaveFig=False, dE=0.01, isplot=True):
        """
        Plot filling vs energy.
        Filling is density of states integrated up to a given energy level.
        It can be considered as a cumulative histogram plot of energy values of
        states in the system.
        """

        cell = self.crystal
        if hasattr(self,'isAFRBZ'):
            X,Y = cell.get_kpoints(dk=0.1, isAFRBZ=True)
            # total number of k points is twice as much in the reduced RBZ zone.
            # Hence multiply by 2
            Nk = 2*X.size # X is a meshgrid
        else:
            X,Y = cell.get_kpoints(dk=0.1)
            Nk =  X.size # X is a meshgrid

        Eall = self.get_Eall(X,Y)
        Emin = Eall.min()
        Emax = Eall.max()

        vfermi = np.vectorize(self.fermiDist)
        def f(x): return sum(sum(vfermi(Eall-x)))/float(Nk)

        elist = np.arange(Emin,Emax,dE)
        vfilling = np.vectorize(f)

        afill = vfilling(elist)
        plt.plot(elist,afill)
        plt.xlabel("Fermi level")
        plt.ylabel("electron density")
        if isSaveFig:
            plt.savefig(self.__name__ + '_filling_vs_fermi_level.png')
        if isplot:
            plt.show()
        return elist,afill

    def get_density(self, E0,Eall,Nvol,Evecs=None):
        return self.filling1(E0,Eall,Nvol)

    #@jit()
    def get_Fermi_level1(self, target_filling: float) -> float:
        """
        This function calculates the fermi level corresponding to
        a given target filling.
        """

        cell = self.crystal
        if hasattr(self,'isAFRBZ'):
            X,Y = cell.get_kpoints(dk=0.1, isAFRBZ=True)
            # total number of k points is twice as much in the reduced RBZ zone.
            # Hence multiply by 2
            Nvol = 2*X.size # X is a meshgrid
        else:
            X,Y = cell.get_kpoints(dk=0.1)
            Nvol =  X.size # X is a meshgrid

        if self.rank > 1:
            Evecs, Eall = get_Evecs(X,Y,self.Ematrix,flatten=False)
        else:
            Evecs = None
            Eall = self.get_Eall(X,Y)

        Emin = Eall.min()
        Emax = Eall.max()

        # use bisection to find the Fermi level
        Emid = (Emin+Emax)/2.
        tol = 0.001
        dn = 5 #initialize
        dn_list = []
        N_iter = 0
        check_dn_ok = False
        while not check_dn_ok:
            density = self.get_density(Emid,Eall,Nvol,Evecs)
            dn = abs(target_filling - density)
            dn_list.append(dn)
            if density > target_filling: #Emid is big
                Emax = Emid
                Emid = (Emin+Emax)/2.
            else:
                Emin = Emid
                Emid = (Emin+Emax)/2.
            N_iter = N_iter + 1
            # make sure dn is less tol three times in a row.
            if N_iter > 3:
                check_dn_ok = bool(np.all(np.array(dn_list[-3:]) < tol))
            if N_iter > 20:
                print("Filling isn't converging for some reason. exiting at N_iter=20")
                break
        return Emid # new fermi level

    def make_cs(self,xx,yy):
        veband = np.vectorize(self.Eband)
        if self.rank == 1: # single band
            Z = veband(xx, yy)
            cs = plt.contour(xx/pi, yy/pi, Z, [self.eFermi], linewidths=3)
        else: # multi band
            for iband in range(0,self.rank):
                Z = veband(xx, yy, iband=iband)
                cs = plt.contour(xx/pi, yy/pi, Z, [self.eFermi], linewidths=3)
        return cs


    def plot_Fermi_surface_contour(self, isSaveFig=False, isExtendedZone=False, dk=0.1, isShow=True):

        # plot all bands
        fig = plt.figure()
        ax = fig.gca()
        if isExtendedZone:
            kmin=-2*pi
            kmax=2*pi
        else:
            kmin=-pi
            kmax=pi

        X = np.arange(kmin, kmax, dk)
        Y = np.arange(kmin, kmax, dk)
        xx, yy = np.meshgrid(X, Y)

        cs = self.make_cs(xx,yy)

        ax.set_xlim(kmin/pi, kmax/pi)
        ax.set_ylim(kmin/pi, kmax/pi)

        ax.set_aspect("equal")

        plt.xlabel("kx/$\pi$")
        plt.ylabel("ky/$\pi$")
        plt.title("Fermi surface")

        #    ax.zaxis.set_major_locator(LinearLocator(10))
        #    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #    fig.colorbar(surf, shrink=0.5, aspect=5)

        # Draw first brilloin zone
        self.crystal.overlay_FBZ(plt)
        if hasattr(self,'isAFRBZ'):
            self.crystal.overlay_RBZ(plt)


        if isSaveFig:
            plt.savefig(self.__name__ + "_fermi_surface.png")
        if isShow:
            plt.show()
        for path in cs.get_paths():
            v = path.vertices
            cx = v[:,0]
            cy = v[:,1]
        return cx,cy


    def plot_bands(self, style='surf', isSaveFig=False, kmin=-pi, kmax=pi):

        # plot all bands
        X = np.arange(kmin, kmax, 0.1)
        Y = np.arange(kmin, kmax, 0.1)
        xx, yy = np.meshgrid(X, Y)


        veband = np.vectorize(self.Eband)  # vectorize
        if self.rank == 1: # single band
            Z = veband(xx, yy)
            if style == 'topview':
                # use pcolor for topview
                fig, ax = plt.subplots()
                c = ax.pcolor( xx/pi, yy/pi, Z, cmap=cm.coolwarm,
                            vmin = np.min(Z), vmax = np.max(Z), shading='auto')
                fig.colorbar(c, ax=ax)

            elif style == 'surf':
                # surface plot
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(
                    xx/pi, yy/pi, Z, rstride=1,
                    cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False
                )
                fig.colorbar(surf, shrink=0.5, aspect=5)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter("%.01f"))

                # for top view in surf mode, do the following
                # ax.view_init(elev=90, azim=-90, roll=0)
            else:
                pass
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            for nb in range(0,self.rank):
                Z   = veband(xx,yy,nb)
                surf = ax.plot_surface(xx/pi, yy/pi, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
            #fig.colorbar(surf, shrink=0.5, aspect=5)
            # somehow figure is cut off. Hence zoom.

        ax.set_xlim(kmin/pi, kmax/pi)
        ax.set_ylim(kmin/pi, kmax/pi)

        lt = [-1, -0.5, 0, 0.5, 1]
        ax.set_xticks(lt)
        ax.set_yticks(lt)
        plt.xlabel("$kx/\pi$")
        plt.ylabel("$ky/\pi$")
        plt.title("Energy bands")

        if isSaveFig:
            plt.savefig(self.__name__ + '_energy_bands.png')
        plt.show()
        return fig

    def plot_bands_along_sym_cuts(self, withdos=False, withhos=False, isSaveFig=False, plot_Emin=-5, plot_Emax=5, num=50):

        veband = np.vectorize(self.Eband)  # vectorize

        ncuts = len(self.crystal.sym_cuts)
        if withdos or withhos:
            nplots = ncuts + 1
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,nplots)
        else:
            nplots = ncuts
            fig, (ax1, ax2, ax3) = plt.subplots(1,nplots)
        axlist = [ax1, ax2, ax3]

        # make points along the cuts
        for i in range(0, ncuts):
            p1,p2 = self.crystal.sym_cuts[i]
            lkx = np.linspace(p1[0], p2[0], num=num)
            lky = np.linspace(p1[1], p2[1], num=num)
            ax = axlist[i]
            ax.axhline(self.eFermi, color='k', ls='--')
            if self.rank == 1: # single band
                Z = veband(lkx,lky)
                ax.plot(Z)
            else: # multi band
                for nb in range(0,self.rank):
                    Z   = veband(lkx,lky,nb)
                    ax.plot(Z)

            ax.set_ylim(plot_Emin,plot_Emax)
            ax.set_xlim(1,len(lkx)-1)
            ax.set_xticks([len(lkx)/2],[])
            # turn off yaxis ticks except for the first plot
            if i != 0:
                ax.set_yticks([],[])
            if i == 0:
                ax.set_ylabel('Energy (eV)')

        if withhos:
            self.spectra.histogram_of_states(ax=ax4,plot_Emin=plot_Emin,plot_Emax=plot_Emax)
            xg=0.12 ; xx=0.31 ; xm=0.50 ; xgg=0.70
        elif withdos:
            print('DoS plot beside energy cuts is not working yet')
            print('use withhos=True instead')
            # TODO
            # we aren't able to change the orientation of
            # of DoS plot
            #self.density_of_states(ax=ax4)
            #xg=0.12 ; xx=0.31 ; xm=0.50 ; xgg=0.70
            pass
        else:
            xg=0.12 ; xx=0.38 ; xm=0.63 ; xgg=0.89
        # indicate symmetry point labels
        fig.text(xg, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        fig.text(xx, 0.075, 'X', fontweight='bold')
        fig.text(xm, 0.075, 'M', fontweight='bold')
        fig.text(xgg, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        # get rid of space between subplots
        plt.subplots_adjust(wspace=0)
        # set figure title
        ttxt=' '.join(self.__name__.split('_'))
        if hasattr(self,'isAFRBZ'):
            tfill=' (filling='+"{:.2f}".format(self.filling)+"/{})".format(self.rank/2)
        else:
            tfill=' (filling='+"{:.2f}".format(self.filling)+"/{})".format(self.rank)
            #tfill=' (filling='+"{:.2f}".format(self.filling)+')'
        ttxt=ttxt + tfill
        fig.text(0.5,0.9, ttxt, horizontalalignment='center')
        if isSaveFig:
            plt.savefig(self.__name__ + '_energy_band_cuts.png')
        plt.show()
        return fig



    #@jit()
    #@njit(float, float64[:,:], int)
    def filling1(self, E0, Eall, Nk):
        """
        calculates filling for a given Fermi level E0
        uses global variables Eall, Nk
        """
        # Eall must be a flat, 1D, numpy array.
        # A given Eall matrix should be flattened as: Eall.flatten()
        if self.rank == 1:
            return sum(sum(self.fermiDist(Eall-E0)))/float(Nk)
        else:
            vfermi = np.vectorize(self.fermiDist)
            return sum(sum(vfermi(Eall-E0)))/float(Nk)


    def filling_multiband(self, E0, Eall, Nvol, Evecs):
        """
        calculates filling for a given Fermi level E0
        uses global variables Eall, Nk
        """
        vl,vc = np.linalg.eig(self.Ematrix(0.,0.)) # get size
        # orbital resolved weight
        sw = np.zeros((Nvol,*vc.shape))
        for ii in range(Nvol):
            vl = Eall[ii, :]
            vc = Evecs[ii,:]
            # create a 2d array with vl provided columnwise and duplicated
            vl2d = np.array([vl,vl])
            vc_conj = vc.conj()
            n_k = vc*vc_conj*self.fermiDist(vl2d - E0)
            sw[ii] = n_k
        orbital_resolved_filling = []
        for iorb in range(self.rank):
            orbital_resolved_filling.append(sw[:,iorb,:].sum()/Nvol )
        return  orbital_resolved_filling


    @staticmethod
    @jit(nopython=True)
    def fermiDist(x):
        "Fermi distribution"
        return 0.5 * (1.0 - np.tanh(x / (2.0 * kT)))


    @staticmethod
    @jit(nopython=True)
    def fermiPrime(x):
        "Fermi distribution derivative"
        g = x / (2.0 * kT)
        denom = 4.0 * kT * (np.cosh(g) ** 2)
        return -1.0 / denom


class CuprateSingleBand(System):
    def __init__(self, filling=None):
        self.crystal = Tetra()
        self.rank = 1
        self.__name__ = 'cuprate_single_band'
        self.filling = self.set_filling(filling)
        self.eFermi = self.get_Fermi_level1(self.filling)
        # Note that these factors should be multiplied by t_ij**2
        self.jfactors = (self.jfact1, self.jfact2, self.jfact3)
        self.hfactors_left = self.get_hlist()
        self.hfactors_right = self.get_hlist_right()
        self.gbasis = [ self.g1, self.g2, self.g3, self.g4 , self.one]
        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.spectra = Spectra(self)
        self.U = 0.5
        self.V = 0.5
        self.Vnn = 0.5

    @staticmethod
    @jit(nopython=True)
    def Eband(kx, ky):
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

    # Functions to be used in current susceptibility calculations
    @staticmethod
    def jfact1(k,q):
        return 4*sin(k[0] + q[0]/2)**2

    @staticmethod
    def jfact2(k,q):
        return 4*sin(k[1] + q[1]/2)**2

    @staticmethod
    def jfact3(k,q):
        return 4*sin(k[0] + k[1] + +q[0]/2 + q[1]/2)**2

    @staticmethod
    # left hand side
    def h1a(k,q):
        # k[0] is kx etc
        return exp(1j*(k[0]+q[0]))

    @staticmethod
    def h1b(k,q):
        # k[0] is kx etc
        return exp(-1j*k[0])

    @staticmethod
    def h2a(k,q):
        # k[0] is kx etc
        return exp(-1j*(k[0]+q[0]))

    @staticmethod
    def h2b(k,q):
        # k[0] is kx etc
        return exp(1j*k[0])

    @staticmethod
    def h3a(k,q):
        # k[1] is ky etc
        return exp(1j*(k[1]+q[1]))

    @staticmethod
    def h3b(k,q):
        # k[1] is ky etc
        return exp(-1j*k[1])

    @staticmethod
    def h4a(k,q):
        # k[1] is ky etc
        return exp(-1j*(k[1]+q[1]))

    @staticmethod
    def h4b(k,q):
        # k[1] is ky etc
        return exp(1j*k[1])

    @staticmethod
    # right hand side
    def h1a_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(1j*(kpq[0]+mq[0])) # h1a(kpq,mq)

    @staticmethod
    def h1b_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(-1j*kpq[0]) # h1b(kpq,mq)

    @staticmethod
    def h2a_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(-1j*(kpq[0]+mq[0])) # h2a(kpq,mq)

    @staticmethod
    def h2b_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(1j*kpq[0]) # h2b(kpq,mq)

    @staticmethod
    def h3a_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(1j*(kpq[1]+mq[1])) # h3a(kpq,mq)

    @staticmethod
    def h3b_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(-1j*kpq[1]) # h3b(kpq,mq)

    @staticmethod
    def h4a_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(-1j*(kpq[1]+mq[1])) # h4a(kpq,mq)

    @staticmethod
    def h4b_right(k,q):
        kpq = (k[0]+q[0], k[1]+q[1]) # k+q
        mq = (-q[0],-q[1])      # -q
        return exp(1j*kpq[1]) # h4b(kpq,mq)


    # current operator goes like J ~ c1*c2 - c2*c1, hence the pairs of h factors.
    #hlist = [ (h1a, h1b), (h2a,h2b), (h3a,h3b), (h4a,h4b) ]
    def get_hlist(self):
        return [ (self.h1a, self.h1b), (self.h2a, self.h2b)]

    def get_hlist_right(self):
        return [ (self.h1a_right, self.h1b_right), (self.h2a_right, self.h2b_right) ]


    # gbasis to be used in extended/generalized RPA calc
    @staticmethod
    def g1(k):
        return cos(k[0])

    @staticmethod
    def g2(k):
        return sin(k[0])

    @staticmethod
    def g3(k):
        return cos(k[1])

    @staticmethod
    def g4(k):
        return sin(k[1])

    @staticmethod
    def one(k):
        return 1

    @staticmethod
    def vmat_direct(qx,qy,U=0., V=0.,Vnn=0.):
        return U + V*( cos(qx) + cos(qy)) + Vnn*2*cos(qx)*cos(qy)

    @staticmethod
    def vmat_direct_gbasis(qx,qy,U=0., V=0.,Vnn=0.):
        len_gbasis=5
        # for now, comment out Vnn
        # in order to account for Vnn, one should add the following extra gbasis functions
        # cos(kx)cos(ky)
        # cos(kx)sin(ky)
        # sin(kx)cos(ky)
        # sin(kx)sin(ky)

        vnz = U + V*( cos(qx) + cos(qy)) #+ Vnn*2*cos(qx)*cos(qy)
        vmat = np.diag([0,0,0,0,vnz])
        return vmat

    @staticmethod
    def vmat_exchange_gbasis(qx,qy,U=0., V=0.,Vnn=0.):
        len_gbasis=5
        vmat = np.diag([V, V, V, V, U])
        return vmat


class CuprateThreeBand(System):
    def __init__(self, filling=None):
        self.crystal = Tetra()
        self.rank = 3
        self.__name__ = 'cuprate_three_band'
        self.filling = self.set_filling(filling)
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.spectra = Spectra(self)
        self.orbital_labels = ['Cu', 'Ox', 'Oy']

    @staticmethod
    def Ematrix(kx,ky):
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
        ic = complex(0,1)
        # 3-band case (Emery model for Cuprates)
        m = np.array([ [ ed, 2.*tpd*np.sin(kx/2.), -2.*tpd*np.sin(ky/2.) ],
                      [ 2.*tpd*np.sin(kx/2.), ex, -4.*tpp*np.sin(kx/2.)*np.sin(ky/2.)],
                      [ -2.*tpd*np.sin(ky/2.), -4.*tpp*np.sin(kx/2.)*np.sin(ky/2.), ey  ]
                    ])
        return m;


    def Eband(self, kx,ky,iband=1):
        """
        make energy bands
        """
        vl,vc = np.linalg.eig(self.Ematrix(kx,ky))
        vl = np.sort(vl)
        return vl[iband]


class CuprateFourBandLCO(System):
    def __init__(self, filling=None):
        self.crystal = Tetra()
        self.rank = 4
        self.__name__ = 'cuprate_four_band_LCO'
        self.filling = self.set_filling(filling)
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.spectra = Spectra(self)

    @staticmethod
    def Ematrix(kx,ky):
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

        m = np.array([ [ t11, np.conj(t21),  np.conj(t31), np.conj(t41) ],
                      [ t21, t22,           np.conj(t32), np.conj(t42) ],
                      [ t31, t32,           t33,          np.conj(t43) ],
                      [ t41, t42,           t43,            t44        ]
                    ])
        return m;

    def Eband(self, kx,ky,iband=1):
        """
        make energy bands
        """
        vl,vc = np.linalg.eig(self.Ematrix(kx,ky))
        vl = np.sort(vl)
        return vl[iband]


class TetraSingleBandDDW(System):
    def __init__(self, filling=None):
        self.crystal = Tetra()
        self.rank = 2
        self.__name__ = 'tetra_single_band_DDW'
        self.isAFRBZ = True
        self.filling = self.set_filling(filling)
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.spectra = Spectra(self)

    @staticmethod
    def Ematrix(kx,ky):
        """
        make energy matrix
        """
        # Reference:
        # Spin and Current Correlation Functions in the d-density Wave State of the Cuprates
        # Tewari et al 2001, https://arxiv.org/abs/cond-mat/0101027
        t=0.3
        tp=0.3*t
        Ek = -2*t*(cos(kx) + cos(ky)) + 4*tp*cos(kx)*cos(ky)
        kQx = kx + pi
        kQy = ky + pi
        EkQ = -2*t*(cos(kQx) + cos(kQy)) + 4*tp*cos(kQx)*cos(kQy)
        W0=0.02
        Wk = W0*0.5*(cos(kx) - cos(ky))
        ic = complex(0,1)

        # basis: c_k, c_{k+Q} where Q=(\pi,\pi)
        m = np.array([
                [ Ek,       1j*Wk ],
                [ -1j*Wk,   EkQ    ]
                ])
        return m

    def Eband(self, kx,ky,iband=1):
        """
        make energy bands
        """
        vl,vc = np.linalg.eig(self.Ematrix(kx,ky))
        vl = np.sort(vl)
        return vl[iband]


class TetraSingleBandSC(System):
    def __init__(self, filling=0.5, D0=0.035, mu=-0.2925):
        self.D0 = D0
        self.crystal = Tetra()
        self.rank = 2
        self.__name__ = 'tetra_single_band_SC'
        self.filling = self.set_filling(filling)
        self.mu = mu
        self.eFermi = self.get_Fermi_level1(self.filling)
        #self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        #self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        #self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.spectra = Spectra(self)
        self.ef_plot_offset=0.

    #@staticmethod
    def Ematrix(self,kx,ky):
        """
        make energy matrix
        """
        # Reference:
        # Spin and Current Correlation Functions in the d-density Wave State of the Cuprates
        # Tewari et al 2001, https://arxiv.org/abs/cond-mat/0101027
        t=0.3
        tp=0.3*t
        Ek = -2*t*(cos(kx) + cos(ky)) + 4*tp*cos(kx)*cos(ky)
        D0=self.D0
        Dk= D0*0.5*(cos(kx) - cos(ky))

        # basis: c_k_spin_up^dagger, c_{-k}_spin_down
        m = np.array([
                [ Ek-self.mu,       Dk ],
                [ Dk,   self.mu -Ek]
                ])
        return m

    def Eband(self, kx,ky,iband=1):
        """
        make energy bands
        """
        vl,vc = np.linalg.eig(self.Ematrix(kx,ky))
        vl = np.sort(vl)
        return vl[iband]

    def get_density(self, E0,Eall,Nvol,Evecs=None):
        density_by_orb = self.filling_multiband(E0=E0,Eall=Eall,Nvol=Nvol, Evecs=Evecs)
        # only return particle density (i.e. first orbital) as opposed to contrubitions from hole density
        return density_by_orb[0]

    def filling_vs_energy(self, isSaveFig=False):
        print("SC model needs a different version of this method. Exiting...")

class HexaSingleBand(System):
    def __init__(self, filling=None):
        self.crystal = Hexa()
        self.rank = 1
        self.__name__ = 'hexa_single_band'
        self.filling = self.set_filling(filling)
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.spectra = Spectra(self)

    @staticmethod
    @jit(nopython=True)
    def Eband(kx, ky):
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


if __name__ == "__main__":
    # supress all warnings. Advanced users might want to undo this.
    warnings.filterwarnings('ignore')
    # default system is Tetra crystal with d-wave symmetry (cuprate)
    cupr = CuprateSingleBand()
    cupr.filling = 0.45
    cupr.plot_bands(isSaveFig=True)
    cupr.filling_vs_energy(isSaveFig=True)
    cupr.plot_Fermi_surface_contour(isSaveFig=True)
    #cupr.plot_chi_vs_q(isSaveFig=True)

    # A hexa example
    hexa = HexaSingleBand()
    hexa.plot_Fermi_surface_contour()
