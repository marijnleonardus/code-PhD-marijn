# author: Marijn Venderbosch
# February 2023

import numpy as np
from scipy.constants import hbar, Boltzmann
from arc import Strontium88, PairStateInteractions

Sr88 = Strontium88()


class AtomicMotion:
    
    def doppler_broadening_tweezer(mass, temperature, trapfrequency, wavelength):
        """
        Parameters
        ----------
        mass : float
            mass of atom in [kg].
        temperature : float
            temperature of atom in tweezer in [K].
        trapfrequency : float
            trap frequency of atom in tweezer in [Hz].
        wavelength : float
            wavelength of interrogation laser in [m].

        Returns
        -------
        sigma_detuning : float
            spread (standard dev.) in detuning 
            as seen by atom as a result of doppler broadening.
        """
        
        # formula from supply. info. https://doi.org/10.1038/s41567-020-0903-z
        variance_momentum = hbar*mass*trapfrequency/(2*np.tanh(hbar*trapfrequency/(2*Boltzmann*temperature)))
        sigma_momentum = np.sqrt(variance_momentum)
        sigma_velocity = sigma_momentum/mass
        
        wavenumber = 2*np.pi/wavelength
        sigma_detuning = wavenumber*sigma_velocity
        
        return variance_momentum, sigma_detuning
    
    
    def trap_frequency_tweezer_radial(mass, waist, trapdepth):
        """
        Parameters
        ----------
        mass : float
            mass of atom in tweezer in [kg].
        waist : float 
            doppler_broadening_tweezer.
        trapdepth : float
            trap depth of optical tweezer in [K].

        Returns
        -------
        radial_trap_freq : float
            radial oscillation frequency of atom in tweezer in [Hz].
        """
        trapdepth_joule = Boltzmann*trapdepth
        radial_trap_freq = np.sqrt(4*trapdepth_joule/(mass*waist**2))
        return radial_trap_freq


class VanDerWaals:
    
    def calculate_c6_coefficients(n, l, j, mj):
        """calculate C6 coefficients using ARC library
        Assumes the quantum numbers are identical for the 2 atoms
        
        parameters
        -------------
        n: integer
            principal quantum number 
        l: integer:
            angular momentum quantum number
        j: integer: 
            total angular momentum quantum number
        mj: integer:
            secondary total angular momentum quantum number
        
        returns
        ----------------
        c6: float
            van der waals interaction coefficient in [h GHz mum^6]
            
        example
        -------------
        So for (61s5s) 3P0 mj=0 state of Sr88:
        - 61, 0, 1, 0, 1
        """
        
        calc = PairStateInteractions(Strontium88(),
                                     n, l, j,
                                     n, l, j,
                                     mj, mj,
                                     s=1)
        theta = 0
        phi = 0
        deltaN = 6
        deltaE = 30e9  # in [Hz]
        
        # getC6perturbatively returns the C6 coefficients
        # expressed in units of h GHz mum^6.
        c6, eigenvectors = calc.getC6perturbatively(theta, phi, 
                                                    deltaN, deltaE,
                                                    degeneratePerturbation=True)
        c6=c6[0]
        return c6
    
    
class Polarizability:
    
    def sr88_3s1(n):
        """
        compute Sr88 polarizability in MHz*cm^2/V^2 for 3S1 state, see Mohan 2022 paper

        Parameters
        ----------
        n : integer
            principal quantum number.

        Returns
        -------
        polarizability : float
            polarizability in [MHz cm^2 V^-2].

        """
        defect = Sr88.getQuantumDefect(n, 0, 1, s=1)
        polarizability = 6.3*1e-11*(n-defect)**7
        return polarizability
         