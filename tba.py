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
import pysnooper

import warnings
warnings.filterwarnings('ignore')

from models import *
from chi import *

matplotlib.use("TkAgg")

# set npool to number of cpus/threads the machine has
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2

ic = np.complex(0, 1.0)

kT = 0.01

def get_max_3D(zxy):
    """
    input is a 3D data tuple like (Z,X,Y)
    find max Z, and corresponding x,y values
    """
    Z, X, Y = zxy
    ind = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    print("max Z is:", Z[ind]," located at qx=",X[ind]," qy=",Y[ind])
    return Z[ind], (X[ind], Y[ind])


class System:
    def __init__(self, model=cuprate_single_band, filling=None):
        self.model = model
        self.crystal = model.crystal
        self.Eband = model.Eband
        self.filling = filling if filling else model.rank-0.55
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.__name__ = model.__name__

    def get_default_eband(self):
        if self.crystal is Tetra:
            return Eband_cuprate

    def make_Eall1(self, X, Y):
        veband = np.vectorize(self.Eband)
        xx,yy = np.meshgrid(X,Y)
        Eall = veband(xx,yy)
        return Eall

    def filling_vs_energy(self, isSaveFig=False):
        """
        Plot filling vs energy.
        Filling is density of states integrated up to a given energy level.
        It can be considered as a cumulative histogram plot of energy values of
        states in the system.
        """

        cell = self.crystal
        dk = 0.1
        # make 1d grids along x, y
        X   = np.arange(cell.pc_kx_min, cell.pc_kx_max, dk)
        Y   = np.arange(cell.pc_ky_min, cell.pc_ky_max, dk)

        Nk = X.size*Y.size;

        Eall = self.make_Eall1(X,Y)
        Emin = Eall.min()
        Emax = Eall.max()

        vfermi = np.vectorize(self.fermiDist)
        def f(x): return sum(sum(vfermi(Eall-x)))/float(Nk)

        elist = np.arange(Emin,Emax,0.01)
        vfilling = np.vectorize(f)

        plt.plot(elist,vfilling(elist))
        plt.xlabel("Fermi level")
        plt.ylabel("electron density")
        if isSaveFig:
            plt.savefig(self.__name__ + '_filling_vs_fermi_level.png')
        plt.show()
        #return elist,vfilling

    @jit()
    def get_Fermi_level1(self, target_filling: float) -> float:
        """
        This function calculates the fermi level corresponding to
        a given target filling.
        """
        cell = self.crystal
        dk = 0.1
        X   = np.arange(cell.pc_kx_min, cell.pc_kx_max, dk)
        Y   = np.arange(cell.pc_ky_min, cell.pc_ky_max, dk)
        Nk = X.size*Y.size

        if self.model.rank == 1:
            Eall = self.make_Eall1(X,Y)
            Emin = Eall.min()
            Emax = Eall.max()
        else:
            Eall = make_Eall(X,Y, self.model.Ematrix)
            Eall = np.sort(Eall)
            Emin = Eall.min()
            Emax = Eall.max()

        # use bisection to find the Fermi level
        tol = 0.001
        Emid = (Emin+Emax)/2.
        tol = 0.01
        dn = 5 #initialize
        N_iter = 0
        while dn>tol and N_iter < 10:
            density = self.filling1(Emid,Eall,Nk)
            #print(density)
            dn = abs(target_filling - density)
            if density > target_filling: #Emid is big
                Emax = Emid
                Emid = (Emin+Emax)/2.
            else:
                Emin = Emid
                Emid = (Emin+Emax)/2.

            #efermi = Emid
            #print "E_fermi = ",efermi
            N_iter = N_iter + 1
        return Emid


    def plot_Fermi_surface_contour(self, isSaveFig=False, kmin=-2*pi, kmax=2*pi):

        # plot all bands
        fig = plt.figure()
        ax = fig.gca()
        X = np.arange(kmin, kmax, 0.1)
        Y = np.arange(kmin, kmax, 0.1)
        xx, yy = np.meshgrid(X, Y)

        veband = np.vectorize(self.Eband)
        Z = veband(xx, yy)
        cs = plt.contour(xx/pi, yy/pi, Z, [self.eFermi], linewidths=3)

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
        self.crystal.overlay_FBZ()

        if isSaveFig:
            plt.savefig(self.__name__ + "_fermi_surface.png")
        plt.show()

        for item in cs.collections:
            for i in item.get_paths():
                v = i.vertices
                cx = v[:,0]
                cy = v[:,1]
        return cx,cy


    def plot_bands1(self, style='surf', isSaveFig=False, kmin=-pi, kmax=pi):

        # plot all bands
        X = np.arange(kmin, kmax, 0.1)
        Y = np.arange(kmin, kmax, 0.1)
        xx, yy = np.meshgrid(X, Y)


        veband = np.vectorize(self.Eband)  # vectorize
        if self.model.rank == 1: # single band
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
            for nb in range(0,self.model.rank):
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

    def histogram_of_states(self,Nk=1000,Nbin=100,ax=None,iband=None,plot_Emin=-5,plot_Emax=5,isSaveFig=False):
        """
        Calculate densitity of states (DOS) via histogram of energies
        """

        cell = self.crystal
        dk = 2*pi/Nk
        X   = np.arange(cell.pc_kx_min, cell.pc_kx_max, dk)
        Y   = np.arange(cell.pc_ky_min, cell.pc_ky_max, dk)
        Nk = X.size*Y.size


        if self.model.rank == 1:
            Eall = self.make_Eall1(X,Y)
        else: # multi band
            Eall = make_Eall(X,Y,self.model.Ematrix)
            Eall.flatten()

        if iband != None:
            eflat = Eall[iband].flatten() # plt.hist needs a flat array it seems
        else:
            eflat = Eall.flatten() # plt.hist needs a flat array it seems

        if ax: # plotting alongside 3d cuts
            ax.axhline(self.eFermi, color='k', ls='--')
            ax.title.set_text('DoS')
            n, bins, patches = ax.hist(eflat, bins=Nbin, density=True, orientation='horizontal')
            ax.set_ylim(plot_Emin,plot_Emax)
            ax.set_yticks([],[])
            ax.set_xticks([],[])
        else: # regular plot
            n, bins, patches = plt.hist(eflat, bins=Nbin, density=True)
            plt.xlabel('Energy levels')
            plt.ylabel('Histogram')
            plt.title('Histogram of states')
            if isSaveFig:
                plt.savefig(self.__name__ + '_histogram_of_states.png')
            plt.show()

    def density_of_states(self,Nk=200, gamma=0.02, ax=None, iband=None,
            plot_Emin=-5, plot_Emax=5, isSaveFig=False, orb_wgt=False, fast=True):
        """
        Calculate densitity of states (DOS) via histogram of energies
        """
        if orb_wgt and fast:
            print("Warning: orb_wgt isn't implemented within the fast algoritm")
            print("Disabling fast algorithm")
            fast = False

        cell = self.crystal
        dk = 2*pi/Nk
        X   = np.arange(cell.pc_kx_min, cell.pc_kx_max, dk)
        Y   = np.arange(cell.pc_ky_min, cell.pc_ky_max, dk)
        Nk = X.size*Y.size


        if self.model.rank == 1:
            Eall = self.make_Eall1(X,Y)
        else: # multi band
            Eall = make_Eall(X,Y,self.model.Ematrix)
            Eall.flatten()
            Evecs = get_Evecs(X,Y,self.model.Ematrix)

        if iband != None:
            eflat = Eall[iband].flatten() # plt.hist needs a flat array it seems
        else:
            eflat = Eall.flatten() # plt.hist needs a flat array it seems

        if fast==True:
            # Use binning or histograms to exponentially speed up DoS calculation
            orb_wgt=False
            Nw = int((plot_Emax-plot_Emin)/gamma)
            nedge = Nw*3 # even 2x Nw seems to be sufficient
            hist,edges = np.histogram(eflat, nedge)
            dE = edges[1]-edges[0]
            nhist = sum(hist)
            ados = np.zeros(Nw)
            ados_orb = np.zeros((self.model.rank,Nw))
            iw = 0
            aomg = np.linspace(plot_Emin,plot_Emax,Nw) # freq array
            for omg in aomg:
                dos = 0
                for ik in range(edges.size-1):
                    Ek = edges[ik]
                    delta = gamma/( (Ek-omg)**2 + gamma**2 )
                    dos = dos + delta*dE*hist[ik]
                ados[iw] = dos/nedge/np.pi
                iw = iw + 1
        else:
            # TODO get rid of naive for loops
            # make it faster
            Nw = int((plot_Emax-plot_Emin)/gamma)
            ados = np.zeros(Nw)
            ados_orb = np.zeros((self.model.rank,Nw))
            iw = 0
            aomg = np.linspace(plot_Emin,plot_Emax,Nw) # freq array
            for omg in aomg:
                dos = 0
                for ik in range(Nk):
                    Evals = Eall[:,ik]
                    Evecmat = Evecs[:,ik].reshape(3,3)
                    # loop over each eigen val and vec
                    for il in range(self.model.rank):
                        Ek = Evals[il]
                        Evec = Evecmat[:,il]
                        delta = gamma/( (Ek-omg)**2 + gamma**2 )
                        # loop over each orbital
                        for iorb in range(self.model.rank):
                            ados_orb[iorb,iw] = ados_orb[iorb,iw] + np.linalg.norm(Evec[iorb])*delta
                        dos = dos + delta
                ados[iw] = dos/Nk/np.pi
                ados_orb[:,iw] = ados_orb[:,iw]/Nk/np.pi
                iw = iw + 1

        if ax: # plotting alongside 3d cuts
            ax.axhline(self.eFermi, color='k', ls='--')
            ax.title.set_text('DoS')
            # TODO
            # plt doesn't have orientation attribute unlike hist plot
            # somehow this needs to be fixed
            # transform trick didn't work
            ax.plot(aomg, ados)
            ax.set_ylim(plot_Emin,plot_Emax)
            ax.set_yticks([],[])
            ax.set_xticks([],[])
        else: # regular plot
            plt.plot(aomg, ados)
            # also plot DoS contribution by each orbital
            if orb_wgt:
                #marker = itertools.cycle(('.','+', 'o', '*'))
                for iorb in range(self.model.rank):
                    if self.model.__name__ == 'cuprate_three_band' and iorb == 2:
                        plt.plot(aomg,ados_orb[iorb,:],marker='+',linestyle='')
                        plt.legend(['Total','Cu-d', 'O-px', 'O-py'])
                    else:
                        plt.plot(aomg,ados_orb[iorb,:])
            plt.xlabel('Energy levels')
            plt.ylabel('Intensity')
            plt.title('Density of states')
            if isSaveFig:
                plt.savefig(self.__name__ + '_density_of_states.png')
            plt.show()

        return aomg,ados,ados_orb

    def plot_bands_along_sym_cuts(self, withdos=False, withhos=False, isSaveFig=False, plot_Emin=-5, plot_Emax=5):

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
            lkx = np.linspace(p1[0], p2[0])
            lky = np.linspace(p1[1], p2[1])
            ax = axlist[i]
            ax.axhline(self.eFermi, color='k', ls='--')
            if self.model.rank == 1: # single band
                Z = veband(lkx,lky)
                ax.plot(Z)
            else: # multi band
                for nb in range(0,self.model.rank):
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
            self.histogram_of_states(ax=ax4,plot_Emin=plot_Emin,plot_Emax=plot_Emax)
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
        ttxt=' '.join(self.model.__name__.split('_'))
        tfill=' (filling='+"{:.2f}".format(self.filling)+')'
        ttxt=ttxt + tfill
        fig.text(0.5,0.9, ttxt, horizontalalignment='center')
        if isSaveFig:
            plt.savefig(self.__name__ + '_energy_band_cuts.png')
        plt.show()
        return fig



    @jit()
    def filling1(self,E0,Eall,Nk):
        """
        calculates filling for a given Fermi level E0
        uses global variables Eall, Nk
        """
        # Eall must be a flat, 1D, numpy array.
        # A given Eall matrix should be flattened as: Eall.flatten()
        if self.model.rank == 1:
            return sum(sum(self.fermiDist(Eall-E0)))/float(Nk)
        else:
            vfermi = np.vectorize(self.fermiDist)
            return sum(sum(vfermi(Eall-E0)))/float(Nk)


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

if __name__ == "__main__":
    # supress all warnings. Advanced users might want to undo this.
    warnings.filterwarnings('ignore')
    # default system is Tetra crystal with d-wave symmetry (cuprate)
    cupr = System()
    cupr.filling = 0.45
    cupr.plot_bands1(isSaveFig=True)
    cupr.filling_vs_energy(isSaveFig=True)
    cupr.plot_Fermi_surface_contour(isSaveFig=True)
    #cupr.plot_chi_vs_q(isSaveFig=True)

    # A hexa example
    hexa = System(hexa_single_band)
    hexa.plot_Fermi_surface_contour()
