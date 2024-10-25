# author: Marijn Venderbosch
# 2023

from scipy.constants import Boltzmann, c, proton_mass
import numpy as np

temperature = 1e-6  # K
mass = 88 * proton_mass
wavelength = 317e-9
pi = np.pi

angular_frequency = 2 * pi * c / wavelength
doppler_broadened = 2 * angular_frequency / c * np.sqrt(2 * np.log(2) * Boltzmann * temperature / mass)

print('omega/2pi is ' + str(np.round(doppler_broadened / 1e3)) + ' kHz')

