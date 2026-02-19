import numpy as np

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
