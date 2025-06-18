from scipy.constants import hbar, pi
import numpy as np

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from utils.units import MHz, h, um
from atom_class import Rydberg

# variables
rabi_freq = 0.4*MHz # rad/s
omega = 2*pi*rabi_freq
R = 3.6*um # [m]

blockade_radius = Rydberg().calculate_rydberg_blockade_radius(omega)
print('blockade radius', np.round(blockade_radius, 2), " um")

interaction_strength_Hz = Rydberg().calculate_interaction_strength(R)
print('interaction', np.round(interaction_strength_Hz/MHz), " MHz")

blockade_error = (hbar*omega)**2/(2*abs(h*interaction_strength_Hz)**2)
print('blockade error ~', np.round(blockade_error, 4))
