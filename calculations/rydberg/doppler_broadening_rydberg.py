# author: Marijn Venderbosch
# 2023

from scipy.constants import Boltzmann, c, proton_mass
import numpy as np


# %% variables

temperature = 1e-6  # K
atom_mass = 88 * proton_mass  # kg
wavelength = 317e-9  # m
pi=np.pi

# %% compute doppler broadening

def compute_doppler_broadening_fwhm(wavelength, temperature):
    
    """source: de Leseluc et. al (2018) doi.org/10.1103/PhysRevA.97.053803
    
    doppler-broadening as a result of spread in velocity
    
    inputs:
    - wavenumber
    - velocity spread (assume Maxwellian distribution)
    
    output
    - doppler-broadening (FWHM) 
    """
    
    wavenumber=2*np.pi/wavelength
    velocity_spread=np.sqrt(Boltzmann*temperature/atom_mass)
    fwhm = 2*np.log(2)*velocity_spread
    return fwhm

doppler_broadening = compute_doppler_broadening(wavelength, temperature)

print('Doppler broadening is ~ 2pi times ' + str(doppler_broadening) + 'Hz')

