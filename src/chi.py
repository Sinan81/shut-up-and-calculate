#!/usr/bin/env python3

# Tight binding approximation of solid state systems

# builtin modules
import time
from multiprocessing import Pool
import os
import pickle
import warnings
import pdb

# extras
# numba results in 30x speed up!!!
from numba import jit
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

import warnings
warnings.filterwarnings('ignore')

#from models import *

#matplotlib.use("TkAgg")

# set npool to number of cpus/threads the machine has
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2

ic = complex(0, 1.0)

#kT = 0.01


class Chi:
    """
    Susceptbility stuff
    """
    def __init__(self, system):
        self.system = system
        self.bare = None
        self.rpa = None
        self.grpa = None
        self.mesh = None
        self.cuts = None
        self.label = 'bare'


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


    #@jit()
    def real_integ_static(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        """
        eFermi: float
        eFermi = self.system.eFermi
        Ek = self.system.Eband(kx, ky)
        Ekq = self.system.Eband(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * self.system.kT
        if abs(Ek - Ekq) < Ecutoff:
            return -self.system.fermiPrime(Ek - eFermi)
        else:
            return -(self.system.fermiDist(Ek - eFermi) - self.system.fermiDist(Ekq - eFermi)) / (Ek - Ekq)


    def calc_cuts_bare(self, num):
        ncuts = len(self.system.crystal.sym_cuts)
        if self.system.rank != 1: # single band
            print('multi band chi not implemented yet. Exiting...')
            return

        if not self.cuts:
            Zcuts = []
            xy_cuts = []
            # make points along the cuts
            for i in range(0, ncuts):
                p1,p2 = self.system.crystal.sym_cuts[i]
                lkx = np.linspace(p1[0], p2[0], num=num)
                lky = np.linspace(p1[1], p2[1], num=num)
                _xy = list(zip(lkx, lky))
                xy_cuts.append(_xy)
                # now zip X,Y so that we can use pool
                # multiprocess pools doesn't work with class methods
                # hence use PPool from pathos module
                tic = time.perf_counter()
                with PPool(npool) as p:
                    Z = p.map(self.real_static, _xy)
                Zcuts.append(Z)
                toc = time.perf_counter()
                print(f"run time: {toc - tic:.1f} seconds")
            self.cuts = Zcuts
            self.cuts_xy = xy_cuts

    def calc_cuts(self, *args, **kwargs):
        return self.calc_cuts_bare(*args, **kwargs)


    def plot_along_sym_cuts(self, num=3, isSaveFig=False, bare=True, rpa=False, grpa=False, ymin=0, ymax=None):
        """
        num: number of points per cut (default 3)
        """

        if self.system.rank != 1: # multiband
            print('multi band chi not implemented yet')
            return

        ncuts = len(self.system.crystal.sym_cuts) # exclude duplicate points
        fig, (ax1, ax2, ax3) = plt.subplots(1,ncuts)
        axlist = [ax1, ax2, ax3]

        ymax_bare = 0
        ymax_rpa  = 0
        ymax_grpa = 0

        if bare:
            self.calc_cuts_bare(num)
            for ax in axlist:
                idx = axlist.index(ax) # get index of item ax
                ax.plot(self.cuts[idx], marker='o', label='bare')
            # enable legend in the last subplot only
            axlist[-1].legend()
            ymax_bare = np.max(self.cuts)

        # don't plot bare sus twice
        if not isinstance(self,ChiCharge):
            self.calc_cuts(num)
            for ax in axlist:
                idx = axlist.index(ax) # get index of item ax
                ax.plot(self.cuts[idx], marker='x', label=self.label)
            # enable legend in the last subplot only
            axlist[-1].legend()
            ymax_rpa = np.max(self.cuts)

        if rpa:
            self.rpa.calc_cuts(num)
            for ax in axlist:
                idx = axlist.index(ax) # get index of item ax
                ax.plot(self.rpa.cuts[idx], marker='x', label="RPA")
            # enable legend in the last subplot only
            axlist[-1].legend()
            ymax_rpa = np.max(self.rpa.cuts)

        if grpa:
            self.grpa.calc_cuts(num)
            for ax in axlist:
                idx = axlist.index(ax) # get index of item ax
                ax.plot(self.grpa.cuts[idx], marker='s', label="GRPA")
            # enable legend in the last subplot only
            axlist[-1].legend()
            ymax_grpa = np.max(self.grpa.cuts)

        ymax_overall = np.max([ymax_bare, ymax_rpa, ymax_grpa])+0.01

        for ax in axlist:
            if ymax is None:
                ymax = ymax_overall
            ax.set_ylim(ymin,ymax) # add a fudge factor
            ax.set_xlim(0,num-1)
            ax.set_xticks([(num-1)/2],[])
        # set ylabel only for first plot
        ax1.set_ylabel('Intensity (unitless)')
        # disable ytics in 2nd to last subplot
        for ax in axlist[1:]:
            ax.set_yticks([],[])

        # indicate symmetry point labels
        fig.text(0.12, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        fig.text(0.38, 0.075, 'X', fontweight='bold')
        fig.text(0.63, 0.075, 'M', fontweight='bold')
        fig.text(0.89, 0.075, '$\mathbf{\Gamma}$', fontweight='bold')
        # get rid of space between subplots
        plt.subplots_adjust(wspace=0)
        # set figure title
        ttxt=' '.join(self.system.__name__.split('_'))
        ttxt='Static charge susceptibility of '+ttxt+' model '+' (filling='+"{:.2f}".format(self.system.filling)+')'
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


    def calc_bare_vs_q(self, Nq=3, show=False, recalc=False, shiftPlot=pi,
            omega=None, plot_zone='full', rpa=None):
        """ calculate susceptibility.
            procedural version is 7x faster
            shiftPlot: set to 'pi' to create a plot around (pi,pi) as opposed to (0.,0.)
        """

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

        self.bare = (Z, X, Y)
        self.data_mesh = self.bare

        if show:
            self.plot_vs_q(Z, X, Y)

        return Z, X, Y

    def calc_vs_q(self, *args, **kwargs):
        print("hello from parent class")
        return calc_bare_vs_q(*args, **kwargs)


    def plot_vs_q(self, style='surf', isSaveFig=False, plot_zone='full', diff_bare=False, Nq=3):

        ttag='susceptibility' + " $\chi(q,\omega=0)$"
        if self.bare is None:
            print('No previous Chi calculation found: self.chi.bare is "None"')
            print('Running self.calc_vs_q()...')
            self.calc_bare_vs_q(plot_zone=plot_zone,Nq=Nq)
        Zbare, _,_ = self.bare

        if self.data_mesh is None:
            self.calc_vs_q(Nq=Nq, plot_zone=plot_zone)
        Z, X, Y = self.data_mesh
        Zplot = Z - Zbare if diff_bare else Z

        # normalise axes
        X = X / pi
        Y = Y / pi

        if style == 'topview':

            fig, ax = plt.subplots()
            c = ax.pcolor( X, Y, Zplot, cmap=cm.coolwarm,
                        vmin = np.min(Zplot), vmax = np.max(Zplot), shading='auto')
            fig.colorbar(c, ax=ax)
        elif style == 'surf':
            # surface plot
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(
                X, Y, Zplot, rstride=1,
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
        plt.title(ttag)

        if isSaveFig:
            plt.savefig(self.system.__name__ + "_susceptibility.png")
        plt.show()

        return plt


class RPA(Chi):
    """
    Susceptbility stuff
    """
    def __init__(self,system):
        super().__init__(system)
        self.label = 'RPA'

    def calc_cuts(self, num):
        ncuts = len(self.system.crystal.sym_cuts)
        # make sure to calculate bare sus first
        if self.cuts is None:
             self.calc_cuts_bare(num)

        Zcuts = []
        for i in range(0,ncuts):
            _xy = self.cuts_xy[i]
            lkx,lky = zip(*_xy)
            chi0 = self.cuts[i]
            z,_,_ = self.get_rpa(chi0, lkx, lky)
            Zcuts.append(z)
        self.cuts= Zcuts


    @staticmethod
    def get_rpa_denominator(chi0, Vmat):
        # sign is '+' for charge sus
        # hence local interaction supress the charge response
        # while non-local interaction can select one are of k-space over another
        return 1 + np.multiply(chi0, Vmat)


    def get_vmat(self, X, Y):
        # X, Y are either mesh or a 1d list of qx and qy values
        def f(qx,qy):
            return self.system.vmat_direct( qx, qy, self.system.U, self.system.V, self.system.Vnn)
        Vmat = np.vectorize(f)
        return Vmat(X, Y)

    def get_rpa(self, chi0, X, Y):
        # X, Y are either mesh or a 1d list of qx and qy values
        def f(qx,qy):
            return self.system.vmat_direct( qx, qy, self.system.U, self.system.V, self.system.Vnn)
        Vmat = np.vectorize(f)
        denom = self.get_rpa_denominator(chi0, Vmat(X,Y))
        Z = np.divide(chi0, denom)
        return (Z, X, Y)


    def calc_vs_q(self, Nq=3, show=False, recalc=False, shiftPlot=pi,
            omega=None, plot_zone='full'):
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
            self.calc_bare_vs_q(Nq=Nq, recalc=recalc, shiftPlot=shiftPlot, omega=omega, plot_zone=plot_zone)

        chi0, X, Y = self.bare
        self.data_mesh = self.get_rpa(chi0,X,Y)


    def get_critical_value(self, q, prange=(0,3), param='U', plot=False):
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


class GeneralizedRPA(Chi):
    """
    Susceptbility stuff
    """
    def __init__(self,system):
        super().__init__(system)
        self.label = 'GRPA'
        self.cuts_gbasis_bare = None
        self.cuts_gbasis_bare_partial = None


    def real_static_gbasis(self, q):
        """
        Real part of zero freq susceptibility integrand
        """
        # TODO reduce the number of integrations by using symmetries
        # a 12x reduction should be possible
        qx, qy = q

        cell = self.system.crystal

        r = dblquad(
                lambda kx, ky: self.real_integ_static_gbasis(kx, ky, qx, qy),
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


    #@jit()
    def real_integ_static_gbasis(self, kx, ky, qx, qy):
        """
        Real part of susceptibility integrand
        cfunc: current susceptibility extra factors
        """
        k = (kx,ky)
        q = (qx,qy)

        if len(self.extra_sus_factor) == 1:
            cfact = self.extra_sus_factor[0](k)
        else:
            gleft, gright = self.extra_sus_factor
            cfact = gleft(k)*gright(k)

        eFermi: float
        eFermi = self.system.eFermi
        Ek = self.system.Eband(kx, ky)
        Ekq = self.system.Eband(kx + qx, ky + qy)
        ##    fermiPrime=0.
        Ecutoff = 1.0 * self.system.kT
        if abs(Ek - Ekq) < Ecutoff:
            return -1*cfact*self.system.fermiPrime(Ek - eFermi)
        else:
            return -1*cfact*(self.system.fermiDist(Ek - eFermi) - self.system.fermiDist(Ekq - eFermi)) / (Ek - Ekq)


    def gbasis_bare(self, _xy):
        """
        calculate bare susceptibility along gbasis:
        chi0^tilde_ij = sum_k G(k) G(k+q) g_i(k) g_j(k)
        where g_i is the ith gbasis function.
        To be used in generalized RPA calcs with ladder diagrams
        """
        Z = ()
        # TODO: prove that gbasis_bare is a symmetric function
        # with respect to matrix diagonal.
        # hence get rid of half of the integrations
        for gfunc_left in self.system.gbasis:
            for gfunc_right in self.system.gbasis:
                self.extra_sus_factor = (gfunc_left, gfunc_right)
                if len(_xy)==1:
                    chi = self.real_static_gbasis(_xy[0])
                else:
                    with PPool(npool) as p:
                        chi = p.map(self.real_static_gbasis, _xy)
                Z = Z + (chi,)
        # return is N_gbasis x N_q
        #     q1  q2 q3
        # g1
        # g2
        # g3

        return np.array(Z)


    def gbasis_bare_partial(self, _xy):
        """
        calculate partial bare susceptibility along gbasis:
        A_i = sum_k G(k) G(k+q) g_i(k)
        where g_i is the ith gbasis function.
        To be used in generalized RPA calcs with ladder diagrams
        """
        # TODO: extract this from gbasis_bare instead
        # get rid of unnecessary calculation
        Z = ()
        for gfunc in self.system.gbasis:
            self.extra_sus_factor = (gfunc,)
            if len(_xy)==1:
                chi = self.real_static_gbasis(_xy[0])
            else:
                with PPool(npool) as p:
                    chi = p.map(self.real_static_gbasis, _xy)
            Z = Z + (chi,)
        return np.array(Z)


    def gbasis_effective_interaction(self,q):
        qx,qy = q
        # vrho_gbasis = V_xc_gbasis -2*V_direct_gbasis
        vrho = self.system.vmat_exchange_gbasis( qx, qy, self.system.U, self.system.V, self.system.Vnn) \
               - 2*self.system.vmat_direct_gbasis( qx, qy, self.system.U, self.system.V, self.system.Vnn)

        qtuple = (q,)
        # TODO save chi_tilde to prevent recalc for different values of V matrix
        chi_tilde = self.gbasis_bare(qtuple).reshape(5,5)
        denom = np.diag(np.ones(5)) - vrho @ chi_tilde
        denom_inv = np.linalg.inv(denom)
        return denom_inv @ vrho


    def gbasis_chi(self,q):
        """Calculate Chi within generalized RPA with infinite sum of
        ladder, bubble, and mixed diagrams with non-local interaction.
        Ref: Collective excitations in the normal state of Cu-O-based superconductors,
        Littlewood etal, 1989
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.39.12371
        """
        qx,qy = q
        qtuple = (q,)

        Amat = self.gbasis_bare_partial(qtuple)
        Gmat = self.gbasis_effective_interaction(q)

        return self.real_static(q) + Amat.T @ Gmat @ Amat


    def get_vrho(self, qx, qy):
        vrho = self.system.vmat_exchange_gbasis( qx, qy, self.system.U, self.system.V, self.system.Vnn) \
               - 2*self.system.vmat_direct_gbasis( qx, qy, self.system.U, self.system.V, self.system.Vnn)
        return vrho


    def gbasis_chi_mesh(self,X,Y):
        """Calculate Chi within generalized RPA with infinite sum of
        ladder, bubble, and mixed diagrams with non-local interaction.
        Ref: Collective excitations in the normal state of Cu-O-based superconductors,
        Littlewood etal, 1989
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.39.12371
        """
        #qx,qy = q
        #qtuple = (q,)
        x = X.reshape(X.size)
        y = Y.reshape(Y.size)
        _xy = list(zip(x, y))

        # This calculation is costly. Hence, check if it already exists
        if self.gbasis_bare_partial_mesh is None:
            # Amat shape is 5 x Nk
            # Do Amat.T to access by k like
            # Amat.T[0] 5x1 Amat corresponding to 1st k.
            Z = self.gbasis_bare_partial(_xy)
            Amat_mesh = Z.T.reshape(X.shape[0],X.shape[1],5)
            self.gbasis_bare_partial_mesh = (Amat_mesh, X, Y)
        #Gmat = self.gbasis_effective_interaction(q)

        # This calculation is costly. Hence, check if it already exists
        if self.gbasis_bare_mesh is None:
            # output shape: Nbasis**2 x Nk**2. hence, re-arrange
            Z = self.gbasis_bare(_xy)
            chi_tilde_mesh = Z.T.reshape(X.shape[0],X.shape[1],5,5)
            self.gbasis_bare_mesh = (chi_tilde_mesh, X, Y)

        # we could use np.einsum here but
        # for loop is more clear and performance is not an issue
        Nx, Ny = X.shape
        chi_delta_mesh = np.empty(X.shape)
        gamma_tilde_mesh = np.empty((Nx,Ny,5,5))
        gamma_mesh = np.empty(X.shape)
        chi_mesh = np.empty(X.shape)
        Amat_mesh, _, _ = self.gbasis_bare_partial_mesh
        chi_tilde_mesh, _, _ = self.gbasis_bare_mesh
        if self.bare is None:
            Z = self.run_npool(X,Y)
            self.bare = (Z, X, Y)
        chi0, Xb, Yb = self.bare
        assert np.allclose(X,Xb)
        assert np.allclose(Y,Yb)

        for ix in range(Nx):
            for iy in range(Ny):
                qx = X[ix][iy]
                qy = Y[ix][iy]
                q = (qx,qy)
                chi_tilde = chi_tilde_mesh[ix][iy]
                Amat = Amat_mesh[ix][iy]
                bare = chi0[ix][iy]

                vrho = self.get_vrho(qx,qy)

                denom = np.diag(np.ones(5)) - vrho @ chi_tilde
                denom_inv = np.linalg.inv(denom)

                Gmat = denom_inv @ vrho # effective interaction in gbasis
                gamma_tilde_mesh[ix][iy] = Gmat # effective interaction in gbasis
                # Enhancement due to interactions: chi_delta = Chi_GRPA - Chi0
                chi_delta = Amat.T @ Gmat @ Amat
                chi_delta_mesh[ix][iy] = chi_delta
                chi_mesh[ix][iy] =  bare + chi_delta
                gamma_mesh[ix][iy] = chi_delta/(bare**2) # effective interaction?


        self.grpa_delta = (chi_delta_mesh, X, Y)
        self.grpa_gamma_mesh = (gamma_mesh, X, Y)
        self.grpa_gamma_tilde_mesh = (gamma_tilde_mesh, X, Y)
        self.grpa = (chi_mesh, X, Y)
        return chi_mesh


    def run_gbasis_chi_cuts(self, num=3):
        lkx = np.linspace(0, pi, num=num)
        lky = np.linspace(0 ,pi, num=num)
        lxy = list(zip(lkx,lky))
        self.gbasis_chi_cuts(lxy)


    def gbasis_chi_cuts(self, lxy, icut):
        """Calculate Chi within generalized RPA with infinite sum of
        ladder, bubble, and mixed diagrams with non-local interaction.
        Ref: Collective excitations in the normal state of Cu-O-based superconductors,
        Littlewood etal, 1989
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.39.12371
        """

        if self.cuts is None:
             self.calc_cuts(num)

        Nq = len(lxy)

        # we could use np.einsum here but
        # for loop is more clear and performance is not an issue
        chi_delta_vec = np.empty(Nq)
        Ngbasis = 5
        gamma_tilde_vec = np.empty((Nq,Ngbasis,Ngbasis))
        gamma_vec = np.empty(Nq)
        chi_vec = np.empty(Nq)
        Amat_vec, _ = self.cuts_gbasis_bare_partial[icut]
        chi_tilde_vec, _ = self.cuts_gbasis_bare[icut]
#        if self.bare is None:
#            Z = self.run_npool(X,Y)
#            self.bare = (Z, X, Y)
#        chi0, Xb, Yb = self.bare
#        assert np.allclose(X,Xb)
#        assert np.allclose(Y,Yb)
        chi0 = self.cuts[icut]

        for iq in range(Nq):
            q = lxy[iq]
            qx,qy = q
            chi_tilde = chi_tilde_vec[iq]
            Amat = Amat_vec[iq]
            bare = chi0[iq]

            vrho = self.get_vrho(qx,qy)

            denom = np.diag(np.ones(Ngbasis)) - vrho @ chi_tilde
            denom_inv = np.linalg.inv(denom)

            Gmat = denom_inv @ vrho # effective interaction in gbasis
            gamma_tilde_vec[iq] = Gmat # effective interaction in gbasis
            # Enhancement due to interactions: chi_delta = Chi_GRPA - Chi0
            chi_delta = Amat.T @ Gmat @ Amat
            chi_delta_vec[iq] = chi_delta
            chi_vec[iq] =  bare + chi_delta
            gamma_vec[iq] = chi_delta/(bare**2) # effective interaction?


        #self.cuts_grpa_delta = (chi_delta_vec, lxy)
        #self.cuts_grpa_gamma = (gamma_vec, lxy)
        #self.cuts_grpa_gamma_tilde = (gamma_tilde_vec, lxy)
        #self.cuts_grpa = (chi_vec, lxy)
        return chi_vec


    def gbasis_chi_vectorized(self,X,Y):

        tic = time.perf_counter()
        def f(qx,qy):
            self.gbasis_chi((qx,qy))

        vchi = np.vectorize(f)
        Z = vchi(X,Y)
        toc = time.perf_counter()
        print(f"run time: {toc - tic:.1f} seconds")



    def calc_cuts_gbasis_bare(self, num):

        ncuts = len(self.system.crystal.sym_cuts)
        Ngbasis = 5
        # output shape: Nbasis**2 x Nk**2. hence, re-arrange
        for icut in range(ncuts):
            lxy = self.cuts_xy[icut]
            Nq=len(lxy)
            Z = self.gbasis_bare(lxy)
            l_chi_tilde = Z.T.reshape(Nq,Ngbasis,Ngbasis)
            self.cuts_gbasis_bare[icut] = (l_chi_tilde, lxy)


    def calc_cuts_gbasis_bare_partial(self, num):
        ncuts = len(self.system.crystal.sym_cuts)
        # Amat shape is 5 x Nk
        # Do Amat.T to access by k like
        # Amat.T[0] 5x1 Amat corresponding to 1st k.
        for icut in range(ncuts):
            lxy = self.cuts_xy[icut]
            Nq = len(lxy)
            Z = self.gbasis_bare_partial(lxy)
            Ngbasis = 5
            l_Amat = Z.T.reshape(Nq,Ngbasis)
            self.cuts_gbasis_bare_partial[icut] = (l_Amat, lxy)


    def calc_cuts(self, num):
        Ncuts = len(self.system.crystal.sym_cuts)
        # make sure to calculate bare sus first
        if self.cuts is None:
             self.calc_cuts_bare(Ncuts, num)

        if self.cuts_gbasis_bare is None:
            self.cuts_gbasis_bare = [None]*Ncuts
            self.calc_cuts_gbasis_bare(num)

        # This calculation is costly. Hence, check if it already exists
        if self.cuts_gbasis_bare_partial is None:
            self.cuts_gbasis_bare_partial = [None]*Ncuts
            self.calc_cuts_gbasis_bare_partial(num)

        Zcuts = []
        for icut in range(len(self.system.crystal.sym_cuts)):
            lxy = self.cuts_xy[icut]
            tic = time.perf_counter()
            chi = self.gbasis_chi_cuts(lxy,icut)
            toc = time.perf_counter()
            print(f"run time: {toc - tic:.1f} seconds")
            Zcuts.append(chi)
        self.cuts = Zcuts


    def calc_vs_q(self, Nq=3, show=False, recalc=False, shiftPlot=pi,
            omega=None, plot_zone='full', rpa=None):
        """
        calculate susceptibility.
        procedural version is 7x faster
            shiftPlot: set to 'pi' to create a plot around (pi,pi) as opposed to (0.,0.)
        """
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
        tic = time.perf_counter()

        Z = self.gbasis_chi_mesh(X,Y)
        self.data_mesh = (Z, X, Y)
        toc = time.perf_counter()

        if show:
            self.plot_vs_q(Z, X, Y)

        return Z, X, Y


class ChiCharge(Chi):
    """
    Charge Susceptibility
    """
    def __init__(self,system):
        # rename original Chi as ChiCharge for completeness
        super().__init__(system)
        self.rpa = RPA(self.system)
        self.grpa = GeneralizedRPA(self.system)
        self.type = 'Charge'


class ChiSpin(Chi):
    """
    Spin Susceptibility
    """
    def __init__(self,system):
        # rename original Chi as ChiCharge for completeness
        super().__init__(system)
        self.rpa = RPA(self.system)
        self.grpa = GeneralizedRPA(self.system)
        self.type = 'Spin'

    @staticmethod
    def get_rpa_denominator(chi0, Vmat):
        # sign is '-' for sping susceptibility
        # hence local interaction can enhance the bare response.
        return 1 - np.multiply(chi0, Vmat)

    def gbasis_effective_interaction(self,q):
        qx,qy = q
        # only exchange interaction contributes to spin susceptbility
        # vsigma_gbasis = V_xc_gbasis
        Vsigma = self.system.vmat_exchange_gbasis( qx, qy, self.system.U, self.system.V, self.system.Vnn)
        qtuple = (q,)
        chi_tilde = self.gbasis_bare(qtuple).reshape(5,5)
        denom = np.diag(np.ones(5)) - Vsigma @ chi_tilde
        denom_inv = np.linalg.inv(denom)
        return denom_inv @ Vsigma


class ChiCurrent(Chi):
    """
    Current Susceptbility
    """
    def __init__(self,system):
        # inherit everything from default chi
        # modify or add methods when necessary
        super().__init__(system)
        self.type = 'Current'


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
        zflat = self.curr_sus_bare(_xy)
        Z = ()
        for z in zflat:
            Z = Z + (np.reshape(zflat, X.shape),)
        self.current_bare_v2 = (Z, X, Y)

        toc = time.perf_counter()
        print(f"run time: {toc - tic:.1f} seconds")
        return Z


    def real_static(self, q) -> float:
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


    #@jit()
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
        Ecutoff = 1.0 * self.system.kT
        if abs(Ek - Ekq) < Ecutoff:
            return -1*cfact*self.system.fermiPrime(Ek - eFermi)
        else:
            return -1*cfact*(self.system.fermiDist(Ek - eFermi) - self.system.fermiDist(Ekq - eFermi)) / (Ek - Ekq)


    def curr_sus_bare(self, _xy):
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
