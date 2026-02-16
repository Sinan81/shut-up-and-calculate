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

