# author: Marijn Venderbosch
# January 2023

import numpy as np
from scipy.constants import electron_mass, c, hbar, alpha, pi
from scipy.constants import epsilon_0 as eps0
from scipy.constants import elementary_charge as e0
import scipy.constants
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

utils_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from units import MHz


a0 = scipy.constants.physical_constants['Bohr radius'][0] # m
hartree_energy = electron_mass*c**2*alpha**2 # J
t = c**2*alpha**2


class Conversion:
    """Collection of functions to do with conversion between different units
    
    Because these function do not need external input, 
    I have chosen to use static methods (can be changed where needed)"""

    @staticmethod
    def rate_to_rdme(Aki, J, En1, En2):
        """
        from Einstein coefficient to Radial density matrix element
        See Eq. 27 from DOI: 10.1119/1.12937
        
        inputs
        - Aki: einstein coefficient [Hz]
        - J: quantum number
        - En1 and En2 [Hz] transition energy/h
        
        returns:
        - radial dipole matrix element in atomic units [a0*e]
        """
        
        rdme = np.sqrt(3*pi*eps0*hbar*c**3/(a0**2*e0*2*np.abs(En1-En2)**3)*(2*J + 1)*Aki)
        return rdme
    
    @staticmethod
    def rdme_to_rate(rdme, J, En1, En2):
        """
        From Radial density matrix element to Einstein coefficient
        
        inputs
        - rdme: matrix element in atomic units
        - J: quantum number
        - En1 and En2 [Hz] transition energy/h
        
        returns:
        - Einstein coefficient in Hz
        """
        
        rate = np.abs(En1 - En2)**3*e0**2/(3*pi*eps0*hbar*c**3*(2*J + 1))*(a0*rdme)**2
        return rate
    
    @staticmethod
    def rate_to_rabi(intensity, linewidth, omega21):
        """
        inputs:
        - intensity [W/m^2]
        - linewidth/Einstein coefficient [Hz]
        - Transition angular freq omega21 [rad/s]
        
        returns:
        - Rabi frequency [Hz]
        """
        rabi_square = 6*pi*c**2*intensity*linewidth/(hbar*omega21**3)
        rabi = rabi_square**(0.5)
        return rabi   
    
    @staticmethod
    def rabi_freq_to_rate(intensity, rabi_freq, omega21):
        """
        inputs:
        - intensity [W/m^2]
        - Rabi freq [Hz]
        - Transition angular freq omega21 [rad/s]
        
        returns:
        - einstein coefficient [Hz]
        """
        
        rate = hbar*omega21**3*rabi_freq**2/(6*pi*c**2*intensity)
        return rate

    @staticmethod
    def intensity_to_electric_field(intensity):
        """
        inputs:
        - intensity [W/m^2]
        
        returns:
        - electric field strength [V/m]
        """
        electric_field_square = 2*intensity/(c*eps0)
        electric_field = np.sqrt(electric_field_square)
        return electric_field

    @staticmethod
    def wavelength_to_freq(wavelength):
        """
        inputs:
        - wavelength [m]
        
        returns:
        - angular freq [rad/s]
        """

        frequency = 2*pi*c/wavelength
        return frequency

    @staticmethod
    def energy_to_wavelength(transition_energy):
        """
        
        Args:
            transition_energy (float) [J]
            wavelength (float) [m]
        
        returns:
            wavelength (float): transition wavelength"""
        
        wavelength = 2*pi*hbar*c/transition_energy
        return wavelength

    @staticmethod
    def compute_rabi_freq(rdme, electric_field):
        """
        inputs:
        - rdme in atomic units [a_0 * e]
        - intensity in [W / m^2]
        
        returns:
        - rabi frequency
        """
        # convert to SI units from atomic
        rdme_se = abs(rdme*a0*e0) # coulom * m

        # compute rabi frequency
        omega = electric_field/hbar*rdme_se
        return omega

    @staticmethod
    def wavenr_from_wavelength(wavelength):
        return 2*pi/wavelength
    
    @staticmethod
    def rabi_to_on_res_saturation(rabi_freq, linewidth):
        """
        Calculate the on-resonance saturation intensity from the Rabi frequency.
        
        Args:
            rabi_freq (float): Rabi frequency in rad/s.
            linewidth (float): Linewidth in rad/s.
        
        Returns:
            float: On-resonance saturation intensity in W/m^2.
        """
        
        s0 = 2*rabi_freq**2/linewidth**2
        return s0
   