# author: Marijn Venderbosch
# 2023

from scipy.constants import Boltzmann, c, proton_mass
import numpy as np


# %% variables

temperature = 1e-6  # K
mass = 88 * proton_mass
wavelength = 317e-9

# %% compute doppler broadening

def compute_doppler_broadening(wavelength, temperature):
    
    angular_freq = 2 * np.pi * c / wavelength   
    fwhm = np.sqrt(8 * Boltzmann * temperature * np.log(2) / (mass * c**2)) * angular_freq
    return fwhm

doppler_broadening = compute_doppler_broadening(wavelength, temperature)

print('Doppler broadening is ~ 2pi times ' + str(doppler_broadening) + 'Hz')

