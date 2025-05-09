# author: Marijn Venderbosch
# February 2023

import numpy as np
from scipy.constants import hbar, Boltzmann, pi
from arc import Strontium88, PairStateInteractions

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from units import GHz, h

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
            c6: (flaot) Van der waals interaction coefficient in [h GHz mum^6]
            
        So for example for (61s5s) 3P0 mj=0 state of Sr88: 61, 0, 1, 0, 1
        """
        
        calc = PairStateInteractions(Strontium88(), n, l, j, n, l, j, mj, mj, s=1)
        theta = 0
        phi = 0
        deltaN = 6
        deltaE = 30e9  # in [Hz]
        
        # getC6perturbatively returns the C6 coefficients
        # expressed in units of h GHz mum^6.
        c6, eigenvectors = calc.getC6perturbatively(theta, phi, deltaN, deltaE, degeneratePerturbation=True)
        c6=c6[0]
        return c6
    
    def calculate_rydberg_blockade_radius(self, omega):
        """calculate R_b

        Args:
            omega (float): angular rabi freq, 2pi times rabi freq. 

        Returns:
            blockade_radius (float): radius in [um]
        """

        C6_Ghz = self.calculate_c6_coefficients(61, 0, 1, 0)
        C6_Hz = C6_Ghz*GHz
        blockade_radius = (abs(h*C6_Hz)/(hbar*omega))**(1/6)
        return blockade_radius
    
    def calculate_interaction_strength(self, R):
        """calculate V_DD(R)

        Args:
            R (float): interatomic distance in [m]

        Returns:
            interaction_strength_Hz: V_DD in [Hz]
        """

        C6_Ghz = self.calculate_c6_coefficients(61, 0, 1, 0)
        C6_Hz = C6_Ghz*GHz
        interaction_strength_Hz = -C6_Hz/R**6
        return interaction_strength_Hz
    
class Polarizability:
    @staticmethod
    def sr88_3s1(n):
        """compute Sr88 polarizability in MHz*cm^2/V^2 for 3S1 state, see Mohan 2022 paper

        Args:
            n (int): principal quantum number.

        Returns:
            polarizability (float): polarizability in [MHz cm^2 V^-2]."""
        
        defect = Sr88.getQuantumDefect(n, 0, 1, s=1)
        polarizability = 6.3*1e-11*(n-defect)**7
        return polarizability
         

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
    