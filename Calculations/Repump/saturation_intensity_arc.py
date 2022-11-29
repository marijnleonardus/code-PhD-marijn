# author Marijn Venderbosch
# october 2022

# %% imports

import numpy as np
from scipy.constants import hbar, c, pi
from arc import *

# %% variables

# user defined
linewidth_689 = 2 * pi * 7.4e3

# source: https://link.aps.org/accepted/10.1103/PhysRevA.92.043418
linewidth_688 = 2 * pi * 4.3e6
linewidth_679 = 2 * pi * 1.4e6
linewidth_707 = 2 * pi * 7e6

# ARC
Sr88 = Strontium88()

#  obtain wavelengths
#  first row: n1, l1, j1.
#  second row: n2, l2, j2,
#  third row: s1, s2

# %% calulations

"""get transition wavelenghts
first row: n1, l1, j1
second row: n2, l2, j2
last row: s1, s2"""

wavelength_689 = Sr88.getTransitionWavelength(5, 0, 0,
                                              5, 1, 1,
                                              0, 1)

wavelength_688 = Sr88.getTransitionWavelength(5, 1, 1,
                                              6, 0, 1,
                                              1, 1)

wavelength_707 = Sr88.getTransitionWavelength(5, 1, 2,
                                              6, 0, 1,
                                              1, 1)

wavelength_679 = Sr88.getTransitionWavelength(5, 1, 0,
                                              6, 0, 1,
                                              1, 1)

# %% functions


def saturation_intensity(linewidth, wavelength):
    """returns saturation intensity in W/m2

    inputs:
    - linewidth in Hz (with 2pi prefactor)
    - wavelength in m"""
    
    return 2 * pi**2 * hbar * c * linewidth / 3 / wavelength**3


sat_int_689 = saturation_intensity(linewidth_689, wavelength_689)
sat_int_679 = saturation_intensity(linewidth_679, wavelength_679)
sat_int_688 = saturation_intensity(linewidth_688, wavelength_688)
sat_int_707 = saturation_intensity(linewidth_707, wavelength_707)


def print_optical_frequency(wavelength):
    """returns optical frequency on wavemeter
    
    inputs:
    - wavelength"""
    
    freq = c / wavelength
    freq_THz = np.round(freq / 1e12, decimals = 5)
    print(str(freq_THz) + " THz")


print_optical_frequency(wavelength_707)
print_optical_frequency(wavelength_679)



