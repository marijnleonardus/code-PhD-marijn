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
wavelength_red_mot = Sr88.getTransitionWavelength(5, 0, 0,  # initial n, l, j
                                          5, 1, 1,          # final n, l, j
                                          0, 1)             # begin and final s

wavelength_688 = Sr88.getTransitionWavelength(5, 1, 1,  # initial n, l, j
                                              6, 0, 1,  # final n, l, j
                                              1, 1)     # begin and final s


# %% functions


def saturation_intensity(linewidth, wavelength):
    """returns saturation intensity in W/m2
    
    inputs:
    - linewidth in Hz (without 2pi prefactor)
    - wavelength in m
    """
    return 2 * pi**2 * hbar * c * linewidth / 3 / wavelength**3

sat_int_red_mot = saturation_intensity(red_mot_linewidth, wavelength_red_mot)



