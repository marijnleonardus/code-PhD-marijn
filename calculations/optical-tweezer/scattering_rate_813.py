# author: Marijn Venderbosch
# january 2023

from scipy.constants import hbar
from classes.optics_class import Optics
from functions.atomic_data_functions import ac_stark_shift_polarizability
from classes.rates import LightAtomInteraction
import numpy as np


# %% variables

# Sr 88
polarizability = 286  # [au], atomic units at 813 nm, source: Madjarov thesis

# tweezer parameters
beam_waist = 0.82e-6  # [m]

# power in 813 AOD beam
power_atoms = 10e-3  # Wz

# %% execution

# compute intensity of AOD moveable tweezer
intensity_813 = Optics.gaussian_beam_intensity(beam_waist, power_atoms)

# compute AC stark shift in Joule
ac_stark_shift_joule = ac_stark_shift_polarizability(polarizability, intensity_813)

# compute AC stark shift in Hz
ac_stark_shift_hz = ac_stark_shift_joule / hbar
print("AC stark shift is: " + str(np.round(ac_stark_shift_hz * 1e-6,1)) + " MHz")




