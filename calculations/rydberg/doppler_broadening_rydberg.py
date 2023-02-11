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
    
    """
    computes doppler-broadening as a result of spread in velocity
    source formulaa; de Leseluc et. al (2018) doi.org/10.1103/PhysRevA.97.053803
    
    inputs:
    - wavenumber
    - velocity spread (assume Maxwellian distribution)
    
    output
    - doppler-broadening (FWHM) 
    """
    
    wavenumber=2*pi/wavelength

    # assuming 1D-maxwell-boltzmann distribution
    velocity_spread=np.sqrt(Boltzmann*temperature/atom_mass)

    # compute standard deviation in doppler-broadening
    sigma=wavenumber*velocity_spread

    # convert sigma to fwhm to match defenition linewidth
    fwhm = 2*np.log(2)*sigma
    return fwhm

doppler_broadening = compute_doppler_broadening_fwhm(wavelength, temperature)
print('Doppler broadening is ~ 2pi times ' + str(doppler_broadening/(2*pi)) + 'Hz')

