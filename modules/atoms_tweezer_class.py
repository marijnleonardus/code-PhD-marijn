# author: Marijn Venderbosch
# july 2023

from scipy.constants import epsilon_0, c, Boltzmann, hbar, pi
import numpy as np


class AtomicMotion:
    @staticmethod
    def doppler_broadening_tweezer(mass, temperature, trapfrequency, wavelength):
        """
        Args:
            mass : (float): mass of atom in [kg].
            temperature : (fl) temperature of atom in tweezer in [K].
            trapfrequency (float): trap frequency of atom in tweezer in [Hz].
            wavelength (float): wavelength of interrogation laser in [m].

        Returns
            sigma_detuning (float): (std dev.) in detuning as seen by atom as a result of doppler broadening.
        """
        
        # formula from supply. info. https://doi.org/10.1038/s41567-020-0903-z
        variance_momentum = hbar*mass*trapfrequency/(2*np.tanh(hbar*trapfrequency/(2*Boltzmann*temperature)))
        sigma_momentum = np.sqrt(variance_momentum)
        sigma_velocity = sigma_momentum/mass
        wavenumber = 2*pi/wavelength
        sigma_detuning = wavenumber*sigma_velocity
        return variance_momentum, sigma_detuning
    
    @staticmethod
    def trap_frequency_radial(mass, waist, trapdepth):
        """
        Args:
            mass (float): mass of atom in tweezer in [kg].
            waist (float): doppler_broadening_tweezer.
            trapdepth (float): trap depth of optical tweezer in [K].

        Returns:
        radial_trap_freq (float): radial oscillation frequency of atom in tweezer in [Hz].
        """

        trapdepth_joule = Boltzmann*trapdepth
        omega_trap_radial = np.sqrt(4*trapdepth_joule/(mass*waist**2))
        trap_freq_rad = omega_trap_radial/(2*pi)
        return trap_freq_rad
    
    @staticmethod
    def trap_frequency_axial(mass, waist, trapdepth):
        """compute axial trap freq

        Args:
            mass (float): mass of atom in tweezer in [kg].
            waist (float): doppler_broadening_tweezer.
            trapdepth (float): trap depth of optical tweezer in [K].

        Returns:
            omega_trap_ax (float): axial trap freq. in [Hz]
        """
        trapdepth_joule = Boltzmann*trapdepth
        omega_trap_ax = np.sqrt(2*trapdepth_joule/(mass*waist**2))
        trap_freq_ax = omega_trap_ax/(2*pi)
        return trap_freq_ax
    

class AtomicCalculations:
    def __init__(self, atomic_unit):
        self.au = atomic_unit

    def ac_stark_shift(self, polarizability, intensity):
        """
        returns AC stark shift

        Args:
            polarizability (float): in atomic units
            intensity (float): unit W/m^2

        Returns:
            AC stark shift: trap depth in J
        """
        shark_shift = polarizability*self.au/(2*c*epsilon_0)*intensity
        return shark_shift
    