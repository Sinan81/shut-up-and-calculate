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
#import pysnooper
import pdb

import warnings
warnings.filterwarnings('ignore')

from models import *
from chi import *
from ebands import *
from util import *

if os.environ.get('MPLBACKEND') is None:
    matplotlib.use("TkAgg")

# set npool to number of cpus/threads the machine has
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2

ic = np.complex(0, 1.0)

kT = 0.01


class System:
    def __init__(self, model=cuprate_single_band, filling=None):
        self.model = model
        self.crystal = model.crystal
        self.Eband = model.Eband
        self.filling = self.set_filling(filling)
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.chic = ChiCharge(self) # static susceptibility chi(omega=0,q)
        self.chij = ChiCurrent(self) # static susceptibility chi(omega=0,q)
        self.chis = Chi(self) # static susceptibility chi(omega=0,q)
        self.__name__ = model.__name__
        self.spectra = Spectra(self)

    def get_default_eband(self):
        if self.crystal is Tetra:
            return Eband_cuprate

    def set_filling(self, filling):
        if not filling:
            if hasattr(self.model,'isAFRBZ'):
                # account for double counting in AF RBZ system
                return self.model.rank/2 - 0.5
            else: # normal system
                return self.model.rank - 0.5
        else:
            return filling

    def make_Eall1(self, xx, yy):
        veband = np.vectorize(self.Eband)
        # xx,yy are meshgrids
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
        if hasattr(self.model,'isAFRBZ'):
            X,Y = cell.get_kpoints(dk=0.1, isAFRBZ=True)
            # total number of k points is twice as much in the reduced RBZ zone.
            # Hence multiply by 2
            Nk = 2*X.size # X is a meshgrid
        else:
            X,Y = cell.get_kpoints(dk=0.1)
            Nk =  X.size # X is a meshgrid

        if self.model.rank == 1:
            Eall = self.make_Eall1(X,Y)
        else:
            Eall = make_Eall(X,Y, self.model.Ematrix)
            Eall = np.sort(Eall)
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
        if hasattr(self.model,'isAFRBZ'):
            X,Y = cell.get_kpoints(dk=0.1, isAFRBZ=True)
            # total number of k points is twice as much in the reduced RBZ zone.
            # Hence multiply by 2
            Nvol = 2*X.size # X is a meshgrid
        else:
            X,Y = cell.get_kpoints(dk=0.1)
            Nvol =  X.size # X is a meshgrid

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
            density = self.filling1(Emid,Eall,Nvol)
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

        veband = np.vectorize(self.Eband)
        if self.model.rank == 1: # single band
            Z = veband(xx, yy)
            cs = plt.contour(xx/pi, yy/pi, Z, [self.eFermi], linewidths=3)
        else: # multi band
            for iband in range(0,self.model.rank):
                Z = veband(xx, yy, iband=iband)
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
        self.crystal.overlay_FBZ(plt)
        if hasattr(self.model,'isAFRBZ'):
            self.crystal.overlay_RBZ(plt)


        if isSaveFig:
            plt.savefig(self.__name__ + "_fermi_surface.png")
        if isShow:
            plt.show()

        for item in cs.collections:
            for i in item.get_paths():
                v = i.vertices
                cx = v[:,0]
                cy = v[:,1]
        return cx,cy


    def plot_bands(self, style='surf', isSaveFig=False, kmin=-pi, kmax=pi):

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
        ttxt=' '.join(self.model.__name__.split('_'))
        if hasattr(self.model,'isAFRBZ'):
            tfill=' (filling='+"{:.2f}".format(self.filling)+"/{})".format(self.model.rank/2)
        else:
            tfill=' (filling='+"{:.2f}".format(self.filling)+"/{})".format(self.model.rank)
            #tfill=' (filling='+"{:.2f}".format(self.filling)+')'
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

class Spectra:
    """
    Spectra stuff
    """
    def __init__(self, system):
        self.system = system
        self.gamma = 0.02
        self.Eall, self.Evecs = self.get_Eigs()
        self.Emin = self.get_Eigs()[0].min() -0.1 # fudge factor
        self.Emax = self.get_Eigs()[0].max() + 0.1

    def get_Eigs(self,Nk=100):
        cell = self.system.crystal
        X,Y = cell.get_kpoints(Nk=Nk)

        if self.system.model.rank == 1:
            Eall = self.system.make_Eall1(X,Y)
        else: # multi band
            Eall = make_Eall(X,Y,self.system.model.Ematrix)
            Eall.flatten()
            Evecs = get_Evecs(X,Y,self.system.model.Ematrix)
        return Eall, Evecs


    def histogram_of_states(self, Nk=200, Nbin=100, ax=None, iband=None, plot_Emin=None, plot_Emax=None, isSaveFig=False):
        """
        Calculate densitity of states (DOS) via histogram of energies
        """
        plot_Emin = self.Emin if plot_Emin is None else plot_Emin
        plot_Emax = self.Emax if plot_Emax is None else plot_Emax

        Eall, _ = self.get_Eigs(Nk=Nk)

        if iband != None:
            eflat = Eall[iband].flatten() # plt.hist needs a flat array it seems
        else:
            eflat = Eall.flatten() # plt.hist needs a flat array it seems

        if ax: # plotting alongside 3d cuts
            ax.axhline(self.system.eFermi, color='k', ls='--')
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


    def spectra_k_w(self, Ek_vals, Ek_vecs_matrix, delta_vals):
        # loop over each eigen val and vec
        dos_k = 0
        ados_orb_k = np.zeros(self.system.model.rank)
        for il in range(self.system.model.rank):
            Ek = Ek_vals[il]
            Evec = Ek_vecs_matrix[:,il]
            # loop over each orbital
            for iorb in range(self.system.model.rank):
                ados_orb_k[iorb] = ados_orb_k[iorb] + np.linalg.norm(Evec[iorb])*delta_vals[il]
        dos_k = np.sum(delta_vals)
        return dos_k, ados_orb_k


    def spectra_w_ik(self, omg, ik):
        #for ik in range(Nk):
        Evals_k = self.Eall[:,ik]
        Evecmat_k = self.Evecs[:,ik].reshape(3,3)
        delta = lambda Ek: self.delta(Ek, omg)
        delta_vals = list(map(delta, Evals_k))
        dos_k, ados_orb_k = self.spectra_k_w(Evals_k, Evecmat_k, delta_vals)
        return (dos_k, ados_orb_k)


    def spectra_w(self, omg):
        dos = 0
        lspectra = lambda ik: self.spectra_w_ik(omg, ik)
        Nk = int(self.Eall.size/self.system.model.rank)
        Nk_list = list(range(Nk))
        #print("Nk_List: ",Nk_list[0:10])
        #pdb.set_trace()
        spectra_vals = list(map(lspectra, Nk_list))
        dos_k_list, ados_orb_k_list = zip(*spectra_vals)
        ados = sum(dos_k_list)/Nk/np.pi
        ados_orb = sum(ados_orb_k_list)/Nk/np.pi
        return (ados, ados_orb)


    def get_spectra_vs_omg(self, plot_Emin, plot_Emax):
        print("plot_Emin is:", plot_Emin)
        print("plot_Emax is:", plot_Emax)

        print("timing orb weight calculation")
        tic = time.perf_counter()
        # TODO get rid of naive for loops
        # make it faster
        Nw = int((plot_Emax-plot_Emin)/self.gamma)
        iw = 0
        aomg = np.linspace(plot_Emin,plot_Emax,Nw) # freq array
        lspectra = lambda omg: self.spectra_w(omg)
        #spectra_vals = list(map(lspectra, aomg))
        with PPool(npool) as p:
            spectra_vals = p.map(self.spectra_w, aomg)
        ados, ados_orb = zip(*spectra_vals)
        toc = time.perf_counter()
        print(f"run time: {toc - tic:.1f} seconds")
        return aomg, ados, ados_orb


    def delta(self, Ek, omg):
        return self.gamma/( (Ek-omg)**2 + self.gamma**2 )


    def get_spectra_vs_omg_via_binning(self, plot_Emin, plot_Emax, iband):
        if iband != None:
            eflat = self.Eall[iband].flatten() # plt.hist needs a flat array it seems
        else:
            eflat = self.Eall.flatten() # plt.hist needs a flat array it seems

        # Use binning or histograms to exponentially speed up DoS calculation
        orb_wgt=False
        print("plot_Emin is:", plot_Emin)
        print("plot_Emax is:", plot_Emax)
        Nw = int((self.Emax-self.Emin)/self.gamma)
        nedge = Nw*3 # even 2x Nw seems to be sufficient
        hist,edges = np.histogram(eflat, nedge)
        dE = edges[1]-edges[0]
        nhist = sum(hist)
        ados = np.zeros(Nw)
        ados_orb = np.zeros((self.system.model.rank,Nw))
        iw = 0
        aomg = np.linspace(plot_Emin,plot_Emax,Nw) # freq array
        for omg in aomg:
            dos = 0
            for ik in range(edges.size-1):
                Ek = edges[ik]
                dos = dos + self.delta(Ek,omg)*dE*hist[ik]
            ados[iw] = dos/nedge/np.pi
            iw = iw + 1
        return aomg, ados


    def density_of_states(self, ax=None, iband=None,
            plot_Emin=None, plot_Emax=None, isSaveFig=False, orb_wgt=False, fast=True):
        """
        Calculate densitity of states (DOS) via histogram of energies
        """
        plot_Emin = self.Emin if plot_Emin is None else plot_Emin
        plot_Emax = self.Emax if plot_Emax is None else plot_Emax


        if orb_wgt and fast:
            print("Warning: orb_wgt isn't implemented within the fast algoritm")
            print("Disabling fast algorithm")
            fast = False

        if fast==True:
            aomg, ados = self.get_spectra_vs_omg_via_binning(plot_Emin=plot_Emin, plot_Emax=plot_Emax, iband=iband)
        else:
            aomg, ados, ados_orb = self.get_spectra_vs_omg(plot_Emin, plot_Emax)

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
                for iorb in range(self.system.model.rank):
                    if self.system.model.__name__ == 'cuprate_three_band' and iorb == 2:
                        plt.plot(aomg,np.array(ados_orb)[:,iorb],marker='+',linestyle='')
                        plt.legend(['Total','Cu-d', 'O-px', 'O-py'])
                    else:
                        plt.plot(aomg, np.array(ados_orb)[:,iorb])
            plt.xlabel('Energy levels')
            plt.ylabel('Intensity')
            plt.title('Density of states')
            if isSaveFig:
                plt.savefig(self.__name__ + '_density_of_states.png')
            plt.show()

        #return aomg,ados,ados_orb



if __name__ == "__main__":
    # supress all warnings. Advanced users might want to undo this.
    warnings.filterwarnings('ignore')
    # default system is Tetra crystal with d-wave symmetry (cuprate)
    cupr = System()
    cupr.filling = 0.45
    cupr.plot_bands(isSaveFig=True)
    cupr.filling_vs_energy(isSaveFig=True)
    cupr.plot_Fermi_surface_contour(isSaveFig=True)
    #cupr.plot_chi_vs_q(isSaveFig=True)

    # A hexa example
    hexa = System(hexa_single_band)
    hexa.plot_Fermi_surface_contour()
