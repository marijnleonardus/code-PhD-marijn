# author: Marijn VEnderbosch
# January 2023
"""to do: clean up some of these classes: some are redundant.
e.g. there is 2 gaussian beam intensity functions now"""

import numpy as np
from scipy.constants import pi


class Optics:
    @staticmethod
    def gaussian_beam_intensity(beam_waist, power):
        """
        inputs:
        - beam_waist [m]
        - power [W]
            
        returns:
        - intensity_0 (max intensity) [W/m^2]
        """
        
        intensity_0 = 2*power/(np.pi*beam_waist**2)
        return intensity_0
    
    @staticmethod
    def cylindrical_gaussian_beam(waist_x, waist_y, power):
        """
        gaussian beam but waist in x and y are not the same (cylindrical)
        
        inputs:
        - beam_waist in x and y directoins [m]
        - power [W]
            
        returns:
        - I0 (max intensity) [W/m^2]
        """
            
        I0 = 2 * power / (np.pi * waist_x * waist_y)
        return I0
    
    @staticmethod
    def gaussian_beam_diffraction_limit(wavelength, numerical_aperture):
        """
        Parameters
        ----------
        wavelength : float, unit [m].
        numerical_aperture : float, unit [].

        Returns
        -------
        Diffraction limited waist w_0 [m]
        assuming airy function can be approximated by a Gaussian beam 
        """
        waist = 0.42 * wavelength / numerical_aperture
        return waist
    
    @staticmethod
    def gaussian_beam_radial(r, beam_waist):
        """
        Parameters
        ----------
        r : float [m]
            positition from center gaussian beam.
        beam_waist : float [m]
            beam waist in center of z direction w_0.

        Returns
        -------
        relative_intensity : float
            I/I0: relative intensity compare to maximum.
        """
        
        relative_intensity = np.exp(-2*r**2 / beam_waist**2)
        return relative_intensity


class GaussianBeam:
    def __init__(self, power, waist):
        self.power = power
        self.waist = waist

    def get_intensity(self):
        """intensity of gaussian beam

        Returns:
            intensity (float): units [W/m^2]
        """
            
        intensity = 2*self.power/pi/self.waist**2
        return intensity
    
    def get_rayleigh_range(self, wavelength):
        """rayleigh range of gaussian beam

        Args:
            wavelength (float): units [m]

        Returns:
            rayleigh_range (float): units [m]
        """
            
        rayleigh_range = pi*self.waist**2/wavelength
        return rayleigh_range
