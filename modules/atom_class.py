# author: Marijn Venderbosch
# February 2023

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, pi
from arc import Strontium88, PairStateInteractions

from utils.units import GHz, h, us, um, MHz

Sr88 = Strontium88()


class Rydberg:
    @staticmethod
    def calculate_c6_coefficients(n, l, j, mj):
        """calculate C6 coefficients using ARC library
        Assumes the quantum numbers are identical for the 2 atoms
        
        Args
            n: (int) principal quantum number 
            l: (int)): angular momentum quantum number
            j: (int): total angular momentum quantum number
            mj: (int)): secondary total angular momentum quantum number
        
        returns
            c6: (flaot) Van der waals interaction coefficient in [Hz mum^6]
            
        So for example for (61s5s) 3P0 mj=0 state of Sr88: 61, 0, 1, 0, 1
        """
        
        calc = PairStateInteractions(Strontium88(), n, l, j, n, l, j, mj, mj, s=1)
        theta = 0
        phi = 0
        deltaN = 6
        deltaE = 30e9  # in [Hz]
        
        # getC6perturbatively returns the C6 coefficients
        # expressed in units of h GHz mum^6.
        c6_GHz, eigenvectors = calc.getC6perturbatively(theta, phi, deltaN, deltaE, degeneratePerturbation=True)
        c6_GHz = c6_GHz[0]
        C6_Hz = c6_GHz*GHz
        return C6_Hz
    
    def calculate_rydberg_blockade_radius(self, omega):
        """calculate R_b

        Args:
            omega (float): angular rabi freq, 2pi times rabi freq. 

        Returns:
            blockade_radius (float): radius in [um]
        """

        C6_Hz = self.calculate_c6_coefficients(61, 0, 1, 0)
        blockade_radius = (abs(h*C6_Hz)/(hbar*omega))**(1/6)
        return blockade_radius
    
    def calculate_interaction_strength(self, R, n):
        """calculate V_DD(R)

        Args:
            R (float): interatomic distance in [m]

        Returns:
            interaction_strength_Hz: V_DD in [Hz]
        """
        R_um = R/um 
        C6_Hz = self.calculate_c6_coefficients(n, 0, 1, 0)
        interaction_strength_Hz = -C6_Hz/R_um**6
        return interaction_strength_Hz


class Sr:
    @staticmethod
    def calc_polarizability_3s1(n):
        """compute Sr88 polarizability in MHz*cm^2/V^2 for 3S1 state, see Mohan 2022 paper

        Args:
            n (int): principal quantum number.

        Returns:
            polarizability (float): polarizability in [MHz cm^2 V^-2]."""
        
        defect = Sr88.getQuantumDefect(n, 0, 1, s=1)
        polarizability = 6.3*1e-11*(n-defect)**7
        return polarizability
    
    @staticmethod
    def calc_rydberg_lifetime(n):
        """computes rydberg lifetime (black body, sponatneous emission) from Madav Mohan paper fig 6.

        Args:
            n (int): principal quantum number.

        Returns:
            lifetime (float): rydberg lifetime in [s]."""
        
        # fit parameters from his paper
        A = 18.84311101
        B = 875.31756526
        
        # quantum defect 
        delta = Sr88.getQuantumDefect(n, 0, 1, s=1)
        
        # compute lifetime rydberg state in s
        lifetime = (A*(n - delta)**(-2) + B*(n - delta)**(-3))**(-1)*1e-6
        return lifetime

    @staticmethod
    def get_rdme(n):
        """computes RDME (radial dipole matrix element) for Sr88 in atomic units
        fucnction is obtained from fitting experimental data of 3P1-r
        which is equivalent for 3P0-r up to a Glebsch-Gordan coefficient

        Args:
            n (int): principal quantun number.

        Returns:
            rdme: (float) radial dipole matrix element [a.u.]"""
        
        # get quantum defect for 3S1 state (l=0, j=1, s=1)
        defect = Sr88.getQuantumDefect(n, 0, 1, s=1)
        
        # get RDME in atomic units
        rdme_au = 1.728*(n - defect)**(-1.5)
        return rdme_au


class AbsorptionImaging:
    @staticmethod
    def compute_cross_section(wavelength):
        """compute cross section

        Args:
            wavelength (float): wavelength in [m]

        Returns:
            cross_section (float): cross section in [m^2]
        """
        cross_section = 3*wavelength**2/(2*pi)
        return cross_section


def main():
    # calc C6 coefficient for (n=61)
    n = 61
    c6_Hz = Rydberg().calculate_c6_coefficients(n, 0, 1, 0)
    print("C6 is", np.round(c6_Hz/GHz), "GHz um^6")

    # calc interaction strength
    R = 3.6*um
    interaction_strength_Hz = Rydberg().calculate_interaction_strength(R, n)
    print('interaction', np.round(interaction_strength_Hz/MHz), " MHz")

    # plot rydberg state lifetimes
    n_grid = np.linspace(40, 100, 11)
    rydberg_state_lifetime_grid = Sr().calc_rydberg_lifetime(n_grid)

    plt.style.use('default')

    fig, ax = plt.subplots()
    ax.scatter(n_grid, rydberg_state_lifetime_grid/us)
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'Rydberg state lifetime [$\mu$s]')
    plt.show()


if __name__ == "__main__":
    main()