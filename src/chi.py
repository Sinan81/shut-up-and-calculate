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

import warnings
warnings.filterwarnings('ignore')

#from models import *

#matplotlib.use("TkAgg")

# set npool to number of cpus/threads the machine has
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2

ic = np.complex(0, 1.0)

kT = 0.01


class Chi:
    """
    Susceptbility stuff
    """
    def __init__(self, system):
        self.bare = None
        self.rpa = None # direct interaction only
        self.grpa = None # with exchange
        self.cuts = None
        self.system = system


    def real_static(self, q):
        """
        Real part of zero freq susceptibility integrand
        """
        # TODO reduce the number of integrations by using symmetries
        # a 12x reduction should be possible
        qx, qy = q

        cell = self.system.crystal

        r = dblquad(
                lambda kx, ky: self.real_integ_static(kx, ky, qx, qy),
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
    def real_integ_static(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        """
        eFermi: float
        eFermi = self.system.eFermi
        Ek = self.system.Eband(kx, ky)
        Ekq = self.system.Eband(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * kT
        if abs(Ek - Ekq) < Ecutoff:
            return -self.system.fermiPrime(Ek - eFermi)
        else:
            return -(self.system.fermiDist(Ek - eFermi) - self.system.fermiDist(Ekq - eFermi)) / (Ek - Ekq)

    @jit()
    def real_integ_static_gbasis(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        cfunc: current susceptibility extra factors
        """
        k = (kx,ky)
        q = (qx,qy)

        if len(self.extra_sus_factor) == 1:
            cfact = self.extra_sus_factor(k)
        else:
            gleft, gright = self.extra_sus_factor
            cfact = gleft(k)*gright(k)

        eFermi: float
        eFermi = self.system.eFermi
        Ek = self.system.Eband(kx, ky)
        Ekq = self.system.Eband(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * kT
        if abs(Ek - Ekq) < Ecutoff:
            return -1*cfact*self.system.fermiPrime(Ek - eFermi)
        else:
            return -1*cfact*(self.system.fermiDist(Ek - eFermi) - self.system.fermiDist(Ekq - eFermi)) / (Ek - Ekq)


    def _gbasis_bare(self, _xy):
        """
        calculate bare current susceptibility
        """
        Z = ()
        # gbasis is diagonal. Hence a single for loop is sufficient
        for gfunc in self.system.gbasis:
            self.extra_sus_factor = (gfunc, gfunc)
            with PPool(npool) as p:
                chi = p.map(self.real_static, _xy)
            Z = Z + (chi,)
            return Z


    def _calc_cuts(self,ncuts,num):
        if not self.cuts:
            Zcuts = []
            # make points along the cuts
            for i in range(0, ncuts):
                p1,p2 = self.system.crystal.sym_cuts[i]
                lkx = np.linspace(p1[0], p2[0], num=num)
                lky = np.linspace(p1[1], p2[1], num=num)
                if self.system.rank == 1: # single band
                    # now zip X,Y so that we can use pool
                    _xy = list(zip(lkx, lky))
                    # multiprocess pools doesn't work with class methods
                    # hence use PPool from pathos module
                    tic = time.perf_counter()
                    with PPool(npool) as p:
                        Z = p.map(self.real_static, _xy)
                    Zcuts.append(Z)
                    toc = time.perf_counter()
                    print(f"run time: {toc - tic:.1f} seconds")
                else: # multi band
                    print('multi band chi not implemented yet')
            self.cuts = Zcuts



    def _plot_individual_cuts(self,ncuts,num,axlist):
        # plot
        for i in range(0, ncuts):
            ax = axlist[i]
            if self.system.rank == 1: # single band
                ax.plot(self.cuts[i], marker='o')
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


    def plot_along_sym_cuts(self, num=3, isSaveFig=False):
        """
        num: number of points per cut (default 3)
        """

        ncuts = len(self.system.crystal.sym_cuts) # exclude duplicate points
        fig, (ax1, ax2, ax3) = plt.subplots(1,ncuts)
        axlist = [ax1, ax2, ax3]

        self._calc_cuts(ncuts,num)
        self._plot_individual_cuts(ncuts,num,axlist)

        # indicate symmetry point labels
        fig.text(0.12, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        fig.text(0.38, 0.075, 'X', fontweight='bold')
        fig.text(0.63, 0.075, 'M', fontweight='bold')
        fig.text(0.89, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        # get rid of space between subplots
        plt.subplots_adjust(wspace=0)
        # set figure title
        ttxt=' '.join(self.system.__name__.split('_'))
        ttxt='Bare susceptibility of '+ttxt +' (filling='+"{:.2f}".format(self.system.filling)+')'
        fig.text(0.5,0.9, ttxt, horizontalalignment='center')
        if isSaveFig:
            plt.savefig(self.system.__name__ + '_chi_cuts.png')
        plt.show()
        return fig


    def run_npool(self,X,Y):

        tic = time.perf_counter()

        # now zip X,Y so that we can use pool
        x = X.reshape(X.size)
        y = Y.reshape(Y.size)
        _xy = list(zip(x, y))

        # multiprocess pools doesn't work with class methods
        # hence use PPool from pathos module
        with PPool(npool) as p:
            chi = p.map(self.real_static, _xy)
        Z = np.reshape(chi, X.shape)
        self.bare = (Z, X, Y)

        toc = time.perf_counter()
        print(f"run time: {toc - tic:.1f} seconds")
        return Z


    def calc_vs_q(self, Nq=3, show=False, recalc=False, shiftPlot=pi,
            omega=None, plot_zone='full', rpa=None):
        """ calculate susceptibility.
            procedural version is 7x faster
            shiftPlot: set to 'pi' to create a plot around (pi,pi) as opposed to (0.,0.)
        """
        import pickle

        if self.system.rank > 1:
            print("Susceptibility calculation isn't implemented for multi orbital systems yet.")
            print("Exiting ...")
            return

        if self.bare != None and recalc == False:
            print("system.chi is already defined. No need to calculate")
            print("force a recalculation with 'recalc=True'")
            return

        # plot all bands
        dq = pi / Nq
        if plot_zone == 'full':
            X = np.arange(-pi + dq + shiftPlot, pi + dq + shiftPlot, dq)
            Y = np.arange(-pi + dq + shiftPlot, pi + dq + shiftPlot, dq)
        elif plot_zone == 'Q1':
            X = np.arange(0 + dq + shiftPlot, pi + dq + shiftPlot, dq)
            Y = np.arange(0 + dq + shiftPlot, pi + dq + shiftPlot, dq)

        X, Y = np.meshgrid(X, Y)

        Z = self.run_npool(X,Y)

        #with open("objs.pkl", "wb") as f:
        #    pickle.dump([Z, X, Y], f)

        if show:
            self.plot_vs_q(Z, X, Y)

        return Z, X, Y


    def plot_vs_q(self, style='surf', isSaveFig=False, plot_zone='full', chi_type='bare'):

        if chi_type == 'bare':
            ttag='Bare susceptibility'
            if self.bare is not None:
                Z, X, Y = self.bare
            else:
                print('No previous Chi calculation found: self.chi.bare is "None"')
                print('Running self.calc_vs_q()...')
                Z, X, Y = self.calc_vs_q(plot_zone=plot_zone)

        if chi_type == 'rpa':
            ttag='RPA susceptibility'
            if self.rpa is not None:
                Z, X, Y = self.rpa
            else:
                print('No previous Chi calculation found: self.chi.bare is "None"')
                print('Running self.calc_vs_q()...')
                Z, X, Y = self.calc_vs_q(rpa='direct_only', plot_zone=plot_zone)

        #matplotlib.use("TkAgg")

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
        plt.title(ttag+" $\chi(q,\omega=0)$")

        if isSaveFig:
            plt.savefig(self.system.__name__ + "_susceptibility.png")
        plt.show()

        # TODO figure out how to save a fig for easly loading later
        #    with open('fig.pkl', 'wb') as f:
        #        pickle.dump([fig], f)

        return plt


    def calc_rpa_vs_q(self, Nq=3, show=False, recalc=False, shiftPlot=pi,
            omega=None, plot_zone='full',rpa_type='direct_only'):
        """ calculate susceptibility.
            procedural version is 7x faster
            shiftPlot: set to 'pi' to create a plot around (pi,pi) as opposed to (0.,0.)
        """
        if self.rpa != None and recalc == False:
            print("system.chi.rpa is already defined. No need to calculate")
            print("force a recalculation with 'recalc=True'")
            return

        if self.bare is None:
            print('No previous bare Chi calculation found.')
            print('Running self.calc_chi_vs_q()...')
            self.calc_vs_q(Nq=Nq, recalc=recalc, shiftPlot=shiftPlot, omega=omega, plot_zone=plot_zone)

        if rpa_type == 'direct_only':
            chi0, X, Y = self.bare
            model = self.system.model
            def f(qx,qy):
                return model.vmat_direct( qx, qy, model.U, model.V, model.Vnn)
            Vmat = np.vectorize(f)
            denom = 1 - np.multiply(chi0, Vmat(X,Y))
            Z = np.divide(chi0, denom)
            self.rpa = (Z, X, Y)


    def rpa_get_critical_value(self, q, prange=(0,3), param='U', plot=False):
        """
        get critical value for a system parameter
        indicating a phase boundary.
        """
        if param == None:
            print("Enter a system parameter, for example: 'param=x.model.U'")

        model = self.system.model
        qx = q[0]
        qy = q[1]
        def f(pval):

            chi_bare = self.real_static(q)
            # if Chi_RPA is diverging, then
            # denominator should be going towards zero
            if   param=='U':
                denom = 1 - chi_bare*model.vmat_direct(qx,qy, pval, model.V, model.Vnn)
            elif param=='V':
                denom = 1 - chi_bare*model.vmat_direct(qx,qy, model.U, pval, model.Vnn)
            elif param=='Vnn':
                denom = 1 - chi_bare*model.vmat_direct(qx,qy, model.U, model.V, pval)
            return denom/chi_bare
        fvec = np.vectorize(f)
        NV = 100
        av = np.linspace(prange[0],prange[1],NV)
        out = np.append(np.empty(0), f(av))
        # zero crossing is where sign changes
        # generally sign changes only once
        # although sometimes re-entrant behvaiour is observed
        # in T vs filling diagrams
        zc = np.where(np.diff(np.sign(out)))[0]
        mid = ( av[zc] + av[zc+1] )/2
        print('Critical value is:',mid)
        if plot:
            plt.plot(av,out)
            plt.axhline(color='r',linestyle=':')
            plt.title("Determining critical parameter value")
            plt.ylabel('$1/\chi$')
            plt.xlabel(param)
            plt.savefig("critical_value.png")
            plt.show()
        return mid, out, av



class ChiCharge(Chi):
    """
    Charge Susceptibility
    """
    def __init__(self,system):
        # rename original Chi as ChiCharge for completeness
        super().__init__(system)


class ChiCurrent(Chi):
    """
    Current Susceptbility
    """
    def __init__(self,system):
        # inherit everything from default chi
        # modify or add methods when necessary
        super().__init__(system)


    def run_npool(self,X,Y):

        tic = time.perf_counter()
        # now zip X,Y so that we can use pool
        x = X.reshape(X.size)
        y = Y.reshape(Y.size)
        _xy = list(zip(x, y))

#        elif sus_type == 'current':
#            z = ()
#            for self.current_sus_factor in self.model.jfactors:
#                with PPool(npool) as p:
#                    chi = p.map(self.real_current_chi_static, _xy)
#                Z = Z + (np.reshape(chi, X.shape),)
#            self.current_bare = (Z,X,Y)
#        elif sus_type == 'current_v2':
        zflat = self._curr_sus_bare(_xy)
        Z = ()
        for z in zflat:
            Z = Z + (np.reshape(zflat, X.shape),)
        self.current_bare_v2 = (Z, X, Y)

        toc = time.perf_counter()
        print(f"run time: {toc - tic:.1f} seconds")
        return Z


    def real_static(self, q):
        """
        Real part of susceptibility integrand
        """
        # TODO reduce the number of integrations by using symmetries
        # a 12x reduction should be possible
        qx, qy = q

        cell = self.system.crystal
        r = dblquad(
                lambda kx, ky: self.real_integ_static(kx, ky, qx, qy),
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
    def real_integ_static(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        cfunc: current susceptibility extra factors
        """
        k = (kx,ky)
        q = (qx,qy)
        if len(self.current_sus_factor) == 1:
            # hand derived current sus factor that's fully real
            cfact = self.current_sus_factor(k,q)
        else:
            # current sus factor = A1A2 -A1B2 -B1A2 + B1B2
            A1,B1 = self.current_sus_factor[0]
            A2,B2 = self.current_sus_factor[1]
            # below, -1 comes from complex constant in current operator definitions: i**2 = -1
            cfact = -1*(A1(k,q)*A2(k,q) - A1(k,q)*B2(k,q) - B1(k,q)*A2(k,q) + B1(k,q)*B2(k,q))
            cfact = np.imag(cfact) if self.cfact_calc == 'imag' else np.real(cfact)

        eFermi: float
        eFermi = self.system.eFermi
        Ek = self.system.Eband(kx, ky)
        Ekq = self.system.Eband(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * kT
        if abs(Ek - Ekq) < Ecutoff:
            return -1*cfact*self.system.fermiPrime(Ek - eFermi)
        else:
            return -1*cfact*(self.system.fermiDist(Ek - eFermi) - self.system.fermiDist(Ekq - eFermi)) / (Ek - Ekq)


    def _curr_sus_bare(self, _xy):
        """
        calculate bare current susceptibility
        """
        Z = ()
        for hleft in self.system.hfactors_left:
            for hright in self.system.hfactors_right:
                self.current_sus_factor = (hleft, hright)
                #print('######')
                for self.cfact_calc  in {'real', 'imag'}:
                    with PPool(npool) as p:
                        chi = p.map(self.real_static, _xy)
                        if self.cfact_calc == 'imag':
                            chi_imag = chi
                        else:
                            chi_real = chi
                ztemp = np.array(chi_real) + 1j*np.array(chi_imag)
                Z = Z + (ztemp,)
                return Z

    def calc_rpa_vs_q(self):
        print("Not implemented for current susceptbility. Exitting ...")
        return
