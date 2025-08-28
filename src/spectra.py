#!/usr/bin/env python3


# builtin modules
import time
import os
import pickle
import warnings

from ebands import *
ncpus = len(os.sched_getaffinity(0))
npool = ncpus if ncpus else 2


class Spectra:
    """
    Spectra stuff
    """
    def __init__(self, system):
        self.system = system # a TBA system as defined in tba.py
        self.gamma = 0.02
        self.Eall, self.Evecs = self.get_Eigs()
        self.Emin = self.get_Eigs()[0].min() -0.1 # fudge factor
        self.Emax = self.get_Eigs()[0].max() + 0.1

    def get_Eigs(self,Nk=100):
        cell = self.system.crystal
        X,Y = cell.get_kpoints(Nk=Nk)

        if self.system.rank == 1:
            Eall = self.system.make_Eall1(X,Y)
            Evecs = None
        else: # multi band
            Eall = make_Eall(X,Y,self.system.Ematrix)
            Eall.flatten()
            Evecs = get_Evecs(X,Y,self.system.Ematrix)
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
        ados_orb_k = np.zeros(self.system.rank)
        for il in range(self.system.rank):
            Ek = Ek_vals[il]
            Evec = Ek_vecs_matrix[:,il]
            # loop over each orbital
            for iorb in range(self.system.rank):
                ados_orb_k[iorb] = ados_orb_k[iorb] + np.linalg.norm(Evec[iorb])*delta_vals[il]
        dos_k = np.sum(delta_vals)
        return dos_k, ados_orb_k


    def spectra_w_ik(self, omg, ik):
        #for ik in range(Nk):
        Evals_k = self.Eall[:,ik]
        Evecmat_k = self.Evecs[:,ik].reshape(self.system.rank,self.system.rank)
        delta = lambda Ek: self.delta(Ek, omg)
        delta_vals = list(map(delta, Evals_k))
        dos_k, ados_orb_k = self.spectra_k_w(Evals_k, Evecmat_k, delta_vals)
        return (dos_k, ados_orb_k)


    def spectra_w(self, omg):
        """ k-integrated spectra """
        dos = 0
        lspectra = lambda ik: self.spectra_w_ik(omg, ik)
        Nk = int(self.Eall.size/self.system.rank)
        Nk_list = list(range(Nk))
        spectra_vals = list(map(lspectra, Nk_list))
        dos_k_list, ados_orb_k_list = zip(*spectra_vals)
        ados = sum(dos_k_list)/Nk/np.pi
        ados_orb = sum(ados_orb_k_list)/Nk/np.pi
        return (ados, ados_orb)


    def spectra_w_vs_k(self, omg):
        """ k-resolved spectra """
        dos = 0
        lspectra = lambda ik: self.spectra_w_ik(omg, ik)
        Nk = int(self.Eall.size/self.system.rank)
        Nk_list = list(range(Nk))
        spectra_vals = list(map(lspectra, Nk_list))
        dos_k_list, ados_orb_k_list = zip(*spectra_vals)
        return (np.array(dos_k_list), np.array(ados_orb_k_list) )


    def plot_spectra_along_cut(self,Emin=-1, Emax=1,kmin=-pi,kmax=pi):
        # cut along ky with kx=pi
        kx = pi
        lky = np.linspace(kmin, kmax, num=200)
        lkx = np.ones(len(lky))*kx

        if self.system.rank == 1:
            Eall = self.system.make_Eall1(lkx,lky)
            Evecs = None
        else: # multi band
            Eall = make_Eall(lkx,lky,self.system.Ematrix)
            Eall.flatten()
            Evecs = get_Evecs(lkx,lky,self.system.Ematrix)
        self.Eall = Eall#-self.system.eFermi
        self.Evecs = Evecs
        lspectra = lambda omg: self.spectra_w_vs_k(omg)
        lomg = np.linspace(Emin,Emax, num=200)
        spectra_vals = list(map(lspectra, lomg))
        # extract tdos for now
        spectra_vals_tdos = []
        for row in spectra_vals:
            spectra_vals_tdos.append(row[0])
        # return a 2d numpy array from list of 1d np arrays.
        data = np.vstack(spectra_vals_tdos)
        im = plt.imshow(data, cmap='viridis',extent=[lky[0]/pi,lky[-1]/pi,lomg[0],lomg[-1]], aspect='auto')
        plt.xlabel("ky/pi with kx=pi")
        plt.ylabel("$\omega$")
        plt.show()


    def plot_spectra_at_fermi_surf(self, orb_resolv=False, isSaveFig=False):

        self.Eall, self.Evecs = self.get_Eigs(Nk=200)
        self.gamma = 0.05
        # total and oribital resolved spectra at fermi surf
        tdos, odos = self.spectra_w_vs_k(omg=self.system.eFermi)
        Nkx = int(np.sqrt(len(tdos)))
        #mtdos = tdos.reshape(Nkx,Nkx)
        #plt.imshow(mtdos)
        #plt.show()
        norbitals = odos.shape[1]
        fsize=6
        fig, axes = plt.subplots(1, norbitals, figsize=(fsize, int(fsize/norbitals)))
        for oind in range(norbitals):
            data = odos.T[oind].reshape(Nkx,Nkx)
            im = axes[oind].imshow(data, cmap='viridis', aspect='equal')
            olabel = self.system.orbital_labels[oind]
            axes[oind].set_title(olabel, fontsize=14, fontweight='bold')
            axes[oind].set_xticks([])
            axes[oind].set_yticks([])
        plt.tight_layout()
        if isSaveFig:
            plt.savefig('spectra_at_fermi_surface.png')
        plt.show()


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
        """ Lorentzian broadenning"""
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
        ados_orb = np.zeros((self.system.rank,Nw))
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
                for iorb in range(self.system.rank):
                    if self.system.__name__ == 'cuprate_three_band' and iorb == 2:
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
