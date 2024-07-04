# author: Marijn Venderbosch
# july 2023

from scipy.constants import epsilon_0, c
import numpy as np


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
    

class TrapFrequencies:

    def trap_freq_radial(self, trapdepth, mass, waist):
        """compute radial trap frequency in rad/s, divide by 2pi to get Hz

        Args:
            trapdepth (float): in unit of J
            mass (float): in kg
            waist (float): in m

        Returns:
            _type_: _description_
        """

        return np.sqrt(4*trapdepth/(mass*waist**2))
    
    def trap_freq_axial(self, trapdepth, mass, rayleigh_range):
        """compute axial trap frequency in rad/s, divide by 2pi to get Hz

        Args:
            trapdepth (float): in units of J
            mass (float): in units of kg
            rayleigh_range (float): in units of m
        """

        return np.sqrt(4*trapdepth/(mass*rayleigh_range**2))