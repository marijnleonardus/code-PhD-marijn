from scipy.constants import hbar, pi
import numpy as np

# local modules
from modules.atom_class import Rydberg
from utils.units import MHz, h, um

# variables
rabi_freq = 1.2*MHz # rad/s
omega = 2*pi*rabi_freq
R = 4*um # [m]
n=61

blockade_radius = Rydberg().calculate_rydberg_blockade_radius(omega)
print('blockade radius', np.round(blockade_radius, 2), " um")

interaction_strength_Hz = Rydberg().calculate_interaction_strength(R, n)
print('interaction', np.round(interaction_strength_Hz/MHz), " MHz")

blockade_error = (hbar*omega)**2/(2*abs(h*interaction_strength_Hz)**2)
print('blockade error ~', np.round(blockade_error, 4))
