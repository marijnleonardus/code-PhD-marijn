# author Marijn Venderbosch
# october 2022

# %% imports

import numpy as np
from scipy.constants import hbar, c, pi
from arc import *

# %% variables

# user defined
red_mot_linewidth = 2 * pi * 7.4e3


# ARC
Sr88 = Strontium88()

#  obtain wavelengths
#  first row: n1, l1, j1.
#  second row: n2, l2, j2,
#  third row: s1, s2
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
    - linewidth in Hz (without 2pi prefactor)
    - wavelength in m
    """
    return 2 * pi**2 * hbar * c * linewidth / 3 / wavelength**3

sat_int_red_mot = saturation_intensity(red_mot_linewidth, wavelength_red_mot)
sat_int_clock = saturation_intensity(0.19, 698e-9)
sat_int_679 = saturation_intensity(2*pi*1.42*10**6,679*10**(-9))



sat_int_689 = saturation_intensity(2 * pi * 7.4e3, wavelength_689)
sat_int_679 = saturation_intensity(2 * pi * 1.4e6, wavelength_679)
sat_int_688 = saturation_intensity(2 * pi * 4.4e6, wavelength_688)
sat_int_707 = saturation_intensity(2 * pi * 9.5e5, wavelength_707)
