# author: Marijn VEnderbosch
# January 2023

import numpy as np
from scipy.constants import pi


class GaussianBeam:
    def __init__(self, power: float, waist: float):
        self.power = power
        self.waist = waist

    def get_intensity(self):
        """intensity of gaussian beam

        Returns:
            intensity (float): units [W/m^2]
        """
            
        max_intensity = 2*self.power/(pi*self.waist**2)
        return max_intensity
    
    def get_rayleigh_range(self, wavelength: float):
        """rayleigh range of gaussian beam

        Args:
            wavelength (float): units [m]

        Returns:
            rayleigh_range (float): units [m]
        """
            
        rayleigh_range = pi*self.waist**2/wavelength
        return rayleigh_range

    def get_diffraction_limit(self, wavelength: float, numerical_aperture: float):
        """ Compute diffraction limited waist for Gaussian that approaches Airy function

        Args:
            wavelength : float, unit [m].
            numerical_aperture : (float)

        Returns:
            diffr_limited_waist (float): beam waist in [m]"""
        diffr_limited_waist = 0.42*wavelength/numerical_aperture
        return diffr_limited_waist
    

class EllipticalGaussianBeam:
    def __init__(self, power: float, waist_x: float, waist_y: float):
        self.power = power
        self.waist_x = waist_x
        self.waist_y = waist_y

    def get_intensity(self):
        """intensity of gaussian beam

        Returns:
            intensity (float): units [W/m^2]
        """
            
        max_intensity = 2*self.power/(pi*self.waist_x*self.waist_y)
        return max_intensity
    