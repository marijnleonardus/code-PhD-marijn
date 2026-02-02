from scipy.constants import hbar, pi
import numpy as np

# append modules dir
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../../modules', '../../../utils'])

from atom_class import Rydberg
from units import MHz, h, um

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
