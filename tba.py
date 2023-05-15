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

from models import *

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
        self.Eband1 = model.Eband
        self.filling = filling if filling else model.rank-0.55
        self.eFermi = self.get_Fermi_level1(self.filling)
        self.chi = None # static susceptibility chi(omega=0,q)
        self.chi_cuts = None
        self.__name__ = model.__name__

    def get_default_eband(self):
        if self.crystal is Tetra:
            return Eband1_cuprate

    def make_Eall1(self, X, Y):
        veband = np.vectorize(self.Eband1)
        xx,yy = np.meshgrid(X,Y)
        Eall = veband(xx,yy)
        return Eall

    def filling_vs_E1(self, isSaveFig=False):
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

        veband = np.vectorize(self.Eband1)
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


        veband = np.vectorize(self.Eband1)  # vectorize
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

    def density_of_states(self,Nk=200, gamma=0.02, ax=None, iband=None, plot_Emin=-5, plot_Emax=5, isSaveFig=False, orb_wgt=False, fast=True):
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

        veband = np.vectorize(self.Eband1)  # vectorize

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


    def _calc_chi_cuts(self,ncuts,num):
        if not self.chi_cuts:
            Zcuts = []
            # make points along the cuts
            for i in range(0, ncuts):
                p1,p2 = self.crystal.sym_cuts[i]
                lkx = np.linspace(p1[0], p2[0], num=num)
                lky = np.linspace(p1[1], p2[1], num=num)
                if self.model.rank == 1: # single band
                    # now zip X,Y so that we can use pool
                    _xy = list(zip(lkx, lky))
                    # multiprocess pools doesn't work with class methods
                    # hence use PPool from pathos module
                    tic = time.perf_counter()
                    with PPool(npool) as p:
                        Z = p.map(self.real_chi_static, _xy)
                    Zcuts.append(Z)
                    toc = time.perf_counter()
                    print(f"run time: {toc - tic:.1f} seconds")
                else: # multi band
                    print('multi band chi not implemented yet')
            self.chi_cuts = Zcuts

    def _plot_individual_chi_cuts(self,ncuts,num,axlist):
        # plot
        for i in range(0, ncuts):
            ax = axlist[i]
            if self.model.rank == 1: # single band
                ax.plot(self.chi_cuts[i], marker='o')
            else: # multi band
                print('multi band chi not implemented yet')

            ax.set_ylim(0,1)
            ax.set_xlim(0,num-1)
            ax.set_xticks([(num-1)/2],[])
            # turn off yaxis ticks except for the first plot
            if i != 0:
                ax.set_yticks([],[])
            if i == 0:
                ax.set_ylabel('Intensity (unitless)')



    def plot_chi_along_sym_cuts(self, isSaveFig=False, num=3):

        ncuts = len(self.crystal.sym_cuts) # exclude duplicate points
        fig, (ax1, ax2, ax3) = plt.subplots(1,ncuts)
        axlist = [ax1, ax2, ax3]

        self._calc_chi_cuts(ncuts,num)
        self._plot_individual_chi_cuts(ncuts,num,axlist)

        # indicate symmetry point labels
        fig.text(0.12, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        fig.text(0.38, 0.075, 'X', fontweight='bold')
        fig.text(0.63, 0.075, 'M', fontweight='bold')
        fig.text(0.89, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        # get rid of space between subplots
        plt.subplots_adjust(wspace=0)
        # set figure title
        ttxt=' '.join(self.model.__name__.split('_'))
        ttxt='Bare susceptibility of '+ttxt +' (filling='+"{:.2f}".format(self.filling)+')'
        fig.text(0.5,0.9, ttxt, horizontalalignment='center')
        if isSaveFig:
            plt.savefig(self.__name__ + '_chi_cuts.png')
        plt.show()
        return fig

    def calc_chi_vs_q(self, Nq=3, show=False, recalc=False, shiftPlot=pi, omega=None, sus_type='charge', plot_zone='full'):
        """ calculate susceptibility.
            procedural version is 7x faster unfortunately
            shiftPlot: set to 'pi' to create a plot around (pi,pi) as opposed to (0.,0.)
        """
        import pickle

        if self.model.rank > 1:
            print("Susceptibility calculation isn't implemented for multi orbital systems yet.")
            print("Exiting ...")
            return

        if self.chi != None and recalc == False:
            print("system.chi is already defined. No need to calculate")
            print("force a recalculation with 'recalc=True'")
            return

        tic = time.perf_counter()
        # plot all bands
        dq = pi / Nq
        if plot_zone == 'full':
            X = np.arange(-pi + dq + shiftPlot, pi + dq + shiftPlot, dq)
            Y = np.arange(-pi + dq + shiftPlot, pi + dq + shiftPlot, dq)
        elif plot_zone == 'Q1':
            X = np.arange(0 + dq + shiftPlot, pi + dq + shiftPlot, dq)
            Y = np.arange(0 + dq + shiftPlot, pi + dq + shiftPlot, dq)

        X, Y = np.meshgrid(X, Y)


        # now zip X,Y so that we can use pool
        x = X.reshape(X.size)
        y = Y.reshape(Y.size)
        _xy = list(zip(x, y))
        if sus_type == 'charge':
            # multiprocess pools doesn't work with class methods
            # hence use PPool from pathos module
            with PPool(npool) as p:
                chi = p.map(self.real_chi_static, _xy)
            Z = np.reshape(chi, X.shape)
        elif sus_type == 'current':
            Z = ()
            for self.current_sus_factor in self.model.jfactors:
                with PPool(npool) as p:
                    chi = p.map(self.real_current_chi_static, _xy)
                Z = Z + (np.reshape(chi, X.shape),)

        #with open("objs.pkl", "wb") as f:
        #    pickle.dump([Z, X, Y], f)

        toc = time.perf_counter()
        print(f"run time: {toc - tic:.1f} seconds")

        if show:
            self.plot_chi_vs_q(Z, X, Y)

        self.chi = (Z, X, Y)

        return Z, X, Y


    def real_chi_static(self, q):
        """
        Real part of susceptibility integrand
        """
        # TODO reduce the number of integrations by using symmetries
        # a 12x reduction should be possible
        qx, qy = q

        cell = self.crystal

        r = dblquad(
                lambda kx, ky: self.real_chi_integ_static(kx, ky, qx, qy),
                cell.integ_xmin,
                cell.integ_xmax,
                cell.gfun,
                cell.hfun,
                # it is ok to comment out the following
                # we specify this to speed calculations up by 2.5x
                # when we set epsabs to 0.1, the precision of the results
                # changed at the most up to third decimal place
                # consistent with 0.1 divided by normalisation factor 4*pi^2
                epsabs=0.1,
            )[0]
        # normalise
        r = r / cell.fbz_area
        print(r)
        return r

    @jit()
    def real_chi_integ_static(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        """
        eFermi: float
        eFermi = self.eFermi
        Ek = self.Eband1(kx, ky)
        Ekq = self.Eband1(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * kT
        if abs(Ek - Ekq) < Ecutoff:
            return -self.fermiPrime(Ek - eFermi)
        else:
            return -(self.fermiDist(Ek - eFermi) - self.fermiDist(Ekq - eFermi)) / (Ek - Ekq)

    def real_current_chi_static(self, q):
        """
        Real part of susceptibility integrand
        """
        # TODO reduce the number of integrations by using symmetries
        # a 12x reduction should be possible
        qx, qy = q

        cell = self.crystal
        r = dblquad(
                lambda kx, ky: self.real_current_chi_integ_static(kx, ky, qx, qy),
                cell.integ_xmin,
                cell.integ_xmax,
                cell.gfun,
                cell.hfun,
                # it is ok to comment out the following
                # we specify this to speed calculations up by 2.5x
                # when we set epsabs to 0.1, the precision of the results
                # changed at the most up to third decimal place
                # consistent with 0.1 divided by normalisation factor 4*pi^2
                epsabs=0.1,
            )[0]
        # normalise
        r = r / cell.fbz_area
        print(r)
        return r


    @jit()
    def real_current_chi_integ_static(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        cfunc: current susceptibility factor
        """
        cfact = self.current_sus_factor
        eFermi: float
        eFermi = self.eFermi
        Ek = self.Eband1(kx, ky)
        Ekq = self.Eband1(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * kT
        if abs(Ek - Ekq) < Ecutoff:
            return -1*cfact((kx,ky), (qx,qy))*self.fermiPrime(Ek - eFermi)
        else:
            return -1*cfact( (kx,ky), (qx,qy))*(self.fermiDist(Ek - eFermi) - self.fermiDist(Ekq - eFermi)) / (Ek - Ekq)


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

    def plot_chi_vs_q(self, style='surf', isSaveFig=False):

        if self.chi is not None:
            Z, X, Y = self.chi
        else:
            print('No previous Chi calculation found: self.chi is not None')
            print('Running self.calc_chi_vs_q()...')
            Z, X, Y = self.calc_chi_vs_q()

        if len(Z) > 1 :
            print("It seem like we're plotting current susceptibility. Will only plot the first element")
            Z = Z[0]

        matplotlib.use("TkAgg")

        # normalise axes
        X = X / pi
        Y = Y / pi

        if style == 'topview':
            fig, ax = plt.subplots()
            c = ax.pcolor( X, Y, Z, cmap=cm.coolwarm,
                        vmin = np.min(Z), vmax = np.max(Z), shading='auto')
            fig.colorbar(c, ax=ax)
        elif style == 'surf':
            # surface plot
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(
                X, Y, Z, rstride=1,
                cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False
            )
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

        # ax.set_zlim(-1.01, 1.01)
        # ax.set_xlim(0, pi)
        # ax.set_ylim(0, pi)

        plt.xlabel("qx/$\pi$")
        plt.ylabel("qy/$\pi$")
        plt.title("Bare static susceptibility $\chi(q,\omega=0)$")

        if isSaveFig:
            plt.savefig(self.__name__ + "_susceptibility.png")
        plt.show()

        # TODO figure out how to save a fig for easly loading later
        #    with open('fig.pkl', 'wb') as f:
        #        pickle.dump([fig], f)

        return plt


if __name__ == "__main__":
    # supress all warnings. Advanced users might want to undo this.
    warnings.filterwarnings('ignore')
    # default system is Tetra crystal with d-wave symmetry (cuprate)
    cupr = System()
    cupr.filling = 0.45
    cupr.plot_bands1(isSaveFig=True)
    cupr.filling_vs_E1(isSaveFig=True)
    cupr.plot_Fermi_surface_contour(isSaveFig=True)
    #cupr.plot_chi_vs_q(isSaveFig=True)

    # A hexa example
    hexa = System(hexa_single_band)
    hexa.plot_Fermi_surface_contour()
