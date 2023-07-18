# author Marijn Venderbosch
# october 2022

# %% imports

import numpy as np
from scipy.constants import hbar, c, pi
from atomphys import Atom


# %% variables

Sr = Atom('Sr')

red_mot_transition = Sr('1S0').to('3P1')

red_mot_linewidth = red_mot_transition.λ.to('m')
#print(red_mot_linewidth)

linewidth_689= 2 * pi * 7.4e3

# %% functions


def saturation_intensity(linewidth, wavelength):
    """returns saturation intensity in W/m2
    
    inputs:
    - linewidth in Hz (without 2pi prefactor)
    - wavelength in m
    """
    return 2 * pi**2 * hbar * c * linewidth / 3 / wavelength**3

red_mot_sat_int = saturation_intensity(2*pi*7.4e3, 
                                       689e-9)




