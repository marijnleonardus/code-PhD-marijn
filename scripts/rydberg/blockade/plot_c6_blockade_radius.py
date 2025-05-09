# author: Marijn Venderbosch
# january 2023

"""script computes Rydberg blockade radii for a low and high estimate 
of Rabi frequencies obtainable as a function of n"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.constants import pi, hbar

# append path with 'modules' dir in parent folder
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from conversion_class import Conversion
from optics_class import GaussianBeam
from rates_class import LightAtomInteraction 
from atom_class import VanDerWaals
from units import mW, MHz, um

## variables
# array of n values
n_start = 55
n_end = 65
n_array = np.linspace(n_start, n_end, n_end-n_start+1)

# rabi freq. calculation
beam_waist = 25*um  # [m]
beam_power = 30*mW
rdmes = LightAtomInteraction.sr88_rdme_value_au(n_array)
RydbergBeam = GaussianBeam(beam_power, beam_waist)
intensity = RydbergBeam.get_intensity()
rabi_frequencies = Conversion.rdme_to_rabi(rdmes, intensity, 1)
print(rabi_frequencies)

# interaction energy
R = 3e-6  # [m]

# C6 coefficients
c6_array = np.array([VanDerWaals.calculate_c6_coefficients(int(n), 0, 1, 0) for n in n_array])

# C6 coeff in terms of h GHz/um^6 to h Hz/um^6
# multiply with 2pi to make it an angular freq. 
c6_Hz = c6_array*1e9*2*pi

# compute blockade raddi for low and high estimates 
blockade_radii = (abs(c6_Hz)/rabi_frequencies)**(1/6)

# arc style adjustment
matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

# plot rabi frequencies
fig0, ax0 = plt.subplots()
ax0.grid()
ax0.scatter(n_array, rabi_frequencies/(2*pi*MHz))
ax0.set_xlabel(r'$n$')
ax0.set_ylabel(r'$\Omega_2/2\pi$ [MHz]')

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
