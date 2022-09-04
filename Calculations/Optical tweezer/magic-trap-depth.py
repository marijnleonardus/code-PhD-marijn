# coding: utf-8

# %% imports

import numpy as np
import matplotlib.pyplot as plt

# %% variables

pi = np.pi

beam_waist = 0.80e-6 # micron
laser_power = 20e-3 # watt

# polarizabilities. Source: Alex' script `TweezerTrapCalculation_With_813.py`
polarizability_1S0_813 = 280.876
polarizability_3P1_mj0_Pi_813nm = 189.9 
polarizability_3P1_mj1_Pi_813nm = 348.71

# %% functions