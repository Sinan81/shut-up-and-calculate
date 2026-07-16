import numpy as np

import matplotlib
import matplotlib.pyplot as plt

class Green():
    """
    Green function
    """
    def __init__(self, system, gamma=None):
        self.system = system # a TBA system as defined in tba.py
        self.gamma = gamma if gamma else 0.02


    def green(self, omg, kx, ky):
        Imat = np.eye(self.system.rank)
        Hmat = self.system.Ematrix(kx,ky)
        return np.linalg.inv( (omg + 1j*self.gamma)*Imat - Hmat )


    def spectra(self, omg, kx, ky, iorb=None):
        Gmat = self.green(omg, kx, ky)
        if iorb is None and hasattr(self.system, 'particle_sector'): #is not None:
            iorb = self.system.particle_sector
        if iorb is None:
            # return total spectra
            # trace is simply a sum of the diagonals
            return -np.imag(np.trace(Gmat))/np.pi
        if type(iorb) is int:
            return -np.imag(np.diagonal(Gmat)[iorb])/np.pi

        # else iorb is iterable
        ssum = 0
        for ii in iorb:
            ssum += -np.imag(np.diagonal(Gmat)[iorb])/np.pi
        return ssum


    def self_energy_static(self,kx,ky,Nq=3,interaction_model='RPA'):
        """
        static self energy calculated via
        RPA charge susceptibility
        Ref: Interacting Electrons: Theory and Computational approaches,
        Martin, Reining, Caperley, 2016
        """
        chic = self.system.chic
        # interaction model options for now:
        #    RPA -> bubble only diagrams
        #   'GRPA' -> generalized RPA with ladder, bubble and mixed diagrams
        if interaction_model == 'RPA':
            chic.rpa.calc_vs_q(Nq=Nq,recalc=True)
            chi_rpa, X, Y = chic.rpa.data_mesh
            Vmat = chic.rpa.get_vmat(X,Y)
            # dielectric function
            epsilon = 1 - chi_rpa*Vmat # elementwise multiplication
            # effective interaction
            Wmat = Vmat/epsilon # elementwise division
            iX = kx - X
            iY = ky - Y
            Eall = self.system.get_Eall(iX,iY)
            vfermi = np.vectorize(self.system.fermiDist)
            occupMat = vfermi(Eall-self.system.eFermi)
            # discrete integration
            # note the elementwise multiplication
            rsum = -sum(sum(Wmat*occupMat))/iX.size
            return rsum
        else:
            print("GRPA isn't implemented yet. exitting ...")
            return None


    def plot_selfenerg_along_sym_cuts(self, isSaveFig=False, plot_Emin=-0.2, plot_Emax=0, num=50, Nq=5):

        #veband = np.vectorize(self.Eband)  # vectorize
        def f(kx,ky):
            return self.self_energy_static(kx,ky,Nq=Nq)

        vfunc = np.vectorize(f)

        ncuts = len(self.system.crystal.sym_cuts)
        nplots = ncuts
        fig, (ax1, ax2, ax3) = plt.subplots(1,nplots)
        axlist = [ax1, ax2, ax3]

        # make points along the cuts
        for i in range(0, ncuts):
            p1,p2 = self.system.crystal.sym_cuts[i]
            lkx = np.linspace(p1[0], p2[0], num=num)
            lky = np.linspace(p1[1], p2[1], num=num)
            ax = axlist[i]
            ax.axhline(self.system.eFermi, color='k', ls='--')
            Z = vfunc(lkx,lky)
            ax.plot(Z)

            ax.set_ylim(plot_Emin,plot_Emax)
            ax.set_xlim(1,len(lkx)-1)
            ax.set_xticks([len(lkx)/2],[])
            # turn off yaxis ticks except for the first plot
            if i != 0:
                ax.set_yticks([],[])
            if i == 0:
                ax.set_ylabel('Energy (eV)')
        xg=0.12 ; xx=0.38 ; xm=0.63 ; xgg=0.89
        # indicate symmetry point labels
        fig.text(xg, 0.075, r'$\mathbf{\Gamma}$', fontweight='bold')
        fig.text(xx, 0.075, 'X', fontweight='bold')
        fig.text(xm, 0.075, 'M', fontweight='bold')
        fig.text(xgg, 0.075, r'$\mathbf{\Gamma}$', fontweight='bold')
        # get rid of space between subplots
        plt.subplots_adjust(wspace=0)
        # set figure title
        ttxt=' '.join(self.system.__name__.split('_'))
        if hasattr(self,'isAFRBZ'):
            tfill=' (filling='+"{:.2f}".format(self.system.filling)+"/{})".format(self.system.rank/2)
        else:
            tfill=' (filling='+"{:.2f}".format(self.system.filling)+"/{})".format(self.system.rank)
            #tfill=' (filling='+"{:.2f}".format(self.filling)+')'
        ttxt='static self Energy' #ttxt + tfill
        fig.text(0.5,0.9, ttxt, horizontalalignment='center')
        if isSaveFig:
            plt.savefig(self.__name__ + '_energy_band_cuts.png')
        plt.show()
        return fig
