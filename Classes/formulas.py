# author: Marijn Venderbosch
# january 2022

import numpy as np
from scipy.constants import c


def beam_intensity(waist, power):
    # gaussian beam
    
    intensity = 2 * power / np.pi / waist**2
    return intensity
