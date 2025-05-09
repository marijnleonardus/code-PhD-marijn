from scipy.constants import hbar, pi
import numpy as np

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from units import MHz, GHz, h
from atom_class import VanDerWaals

# variables
rabi_freq = 7*MHz # rad/s
omega = 2*pi*rabi_freq
R = 3.6 # um

blockade_radius = VanDerWaals().calculate_rydberg_blockade_radius(omega)
print('blockade radius', np.round(blockade_radius, 2), " um")

interaction_strength_Hz = VanDerWaals().calculate_interaction_strength(R)
print('interaction', np.round(interaction_strength_Hz/MHz), " MHz")

blockade_error = (hbar*omega)**2/(2*abs(h*interaction_strength_Hz)**2)
print('blockade error ~', np.round(blockade_error, 4))
