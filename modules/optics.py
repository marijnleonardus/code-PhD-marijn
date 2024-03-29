# author: Marijn VEnderbosch
# January 2023

import numpy as np


class Optics:
    
    def gaussian_beam_intensity(beam_waist, power):
        """
        inputs:
        - beam_waist [m]
        - power [W]
            
        returns:
        - I0 (max intensity) [W/m^2]
        """
        
        I0 = 2 * power / (np.pi * beam_waist**2)
        return I0
    
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
        
        relative_intensity = np.exp(-2 * r**2 / beam_waist**2)
        return relative_intensity