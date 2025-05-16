# author: Marijn Venderbosch
# january 2023

"""script computes Rydberg blockade radii for a low and high estimate 
of Rabi frequencies obtainable as a function of n"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.constants import pi

# append path with 'modules' dir in parent folder
import sys
import os
from scipy.constants import pi

script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from optics_class import GaussianBeam
from laser_class import AtomLightInteraction 
from atom_class import Rydberg
from units import mW, MHz, um

## variables
# array of n values
n_start = 40
n_end = 80
n_array = np.linspace(n_start, n_end, n_end-n_start+1)

# rabi freq. calculation
beam_waist = 18*um  # [m]
beam_power = 30*mW
RydbergBeam = GaussianBeam(beam_power, beam_waist)
intensity = RydbergBeam.get_intensity()
rabi_freqs = AtomLightInteraction.calc_rydberg_rabi_freq(n_array, intensity, j_e=1)
rabi_frequencies_enhanced = np.sqrt(2)*rabi_freqs

# interaction energy
R = 3e-6  # [m]

# C6 coefficients
c6_array = np.array([Rydberg.calculate_c6_coefficients(int(n), 0, 1, 0) for n in n_array])
print(c6_array[3])

# compute blockade raddi for low and high estimates 
blockade_radii = Rydberg().calculate_rydberg_blockade_radius(rabi_freqs)

# arc style adjustment
matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

# plot rabi frequencies
fig0, ax0 = plt.subplots()
ax0.grid()
ax0.scatter(n_array, rabi_freqs/(2*pi*MHz), label=r'$\Omega/2\pi$ [MHz]')
ax0.scatter(n_array, rabi_frequencies_enhanced/(2*pi*MHz), label=r'$\Omega_2/2\pi$ [MHz]')
ax0.legend()
ax0.set_xlabel(r'$n$')
ax0.set_ylabel('Rabi frequency')

# plot C6 coefficient vs n
fig1, ax1 = plt.subplots()
ax1.grid()
ax1.scatter(n_array, abs(c6_array))
ax1.set_yscale('log')
ax1.set_xlabel('$n$')
ax1.set_ylabel(r'$|C_6|$ [GHz $\mu$m$^6$]')

# plot blockade radii vs n
fig2, ax2 = plt.subplots()
ax2.grid()
ax2.scatter(n_array, blockade_radii, label=rf'$w_0={beam_waist/um}$ $\mu$m, $P={beam_power/mW}$ mW')
ax2.set_xlabel(r'$n$')
ax2.set_ylabel(r'Blockade radius [$\mu$m]')
ax2.legend()

plt.show()
