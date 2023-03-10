# author: Marijn Venderbosch
# january 2023

"""script computes Rydberg blockade radii for a low and high estimate 
of Rabi frequencies obtainable as a function of n"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from modules.conversion_class import Conversion
from modules.optics_class import Optics
from modules.rates_class import LightAtomInteraction 
from modules.atom_class import VanDerWaals

# %% variables

# array of n values
n_start = 50
n_end = 100
n_array = np.linspace(n_start, n_end, n_end-n_start+1)

# min. and max. waists to consider
waist_high = 200e-6  # [m]
waist_low = 20e-6  # [m]

# low and high estimates of available power
power_low = 20e-3  # [W]
power_high = 100e-3  # [W]

# interatomic distance
R=4e-6  # [m]

# %% import data

# calculate light intensities
intensity_low = Optics.cylindrical_gaussian_beam(waist_low, waist_high, power_low)
intensity_high = Optics.gaussian_beam_intensity(waist_low, power_high)

# import RDME values
rdmes = LightAtomInteraction.sr88_rdme_value_au(n_array)

# compute rabi frequencies from RDME values
rabi_freqs_low = Conversion.rdme_to_rabi(rdmes, intensity_low, 1)
rabi_freqs_high = Conversion.rdme_to_rabi(rdmes, intensity_high, 1)

# %% compute C6 coefficients and Rydberg blockade radius

# C6 coefficients
c6_array = []

for n in n_array:
    # n has to be int for ARC
    n = int(n)
    
    # get C6 coefficients
    c6 = VanDerWaals.calculate_c6_coefficients(n, 0, 1, 0)
    c6_array.append(c6)

c6_array = np.array(c6_array)

# Blockade radius

# C6 coeff in terms of GHz/um^6 to Hz/um^6
c6_Hz = c6_array*1e9

# compute blockade raddi for low and high estimates 
blockade_radii_low = (abs(c6_Hz)/rabi_freqs_low)**(1/6)
blockade_radii_high = (abs(c6_Hz)/rabi_freqs_high)**(1/6)

# %% plotting

matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

# plot C6 coefficient vs n
fig1, ax1 = plt.subplots()

ax1.grid()
ax1.plot(n_array, abs(c6_array))

ax1.set_yscale('log')
ax1.set_xlabel('$n$')
ax1.set_ylabel('$|C_6|$ [GHz $\mu$m$^6$]')

# plot blockade radii vs n
fig2, ax2 = plt.subplots()
ax2.grid()

ax2.plot(n_array, blockade_radii_low, label=f'$w_x={waist_high*1e6}$ $\mu$m,$w_y={waist_high*1e6}$ $\mu$m, $P={power_low*1e3}$ mW')
ax2.plot(n_array, blockade_radii_high, label=f'$w_x={waist_low*1e6}$ $\mu$m,$w_y={waist_low*1e6}$ $\mu$m, $P={power_high*1e3}$ mW')

ax2.set_xlabel('$n$')
ax2.set_ylabel('Blockade radius [$\mu$m]')
ax2.legend()

plt.show()
