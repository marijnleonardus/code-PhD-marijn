# author: Marijn Venderbosch
# january 2023

import numpy as np
from numpy import genfromtxt
from functions.atomic_data_functions import calculate_c6_coefficients
import matplotlib.pyplot as plt
import matplotlib


# %% variables

# array of n values
n_values_fit = genfromtxt('calculations/rydberg/data/n_values_plot.csv', delimiter=',')

# remove n=19 to n=39: they were used to fit the experimental data which went from n=19 
# to n=40, but these values are not in the relevant range for us
indices_to_delete = np.where(n_values_fit < 40)
n_values_plot = np.delete(n_values_fit, indices_to_delete)

# rabi freq vs n unit is in Hz
rabi_freqs_fromimport = genfromtxt('calculations/rydberg/data/rabi_freq_vs_n.csv', delimiter=',')
rabi_freqs = np.delete(rabi_freqs_fromimport, indices_to_delete)

# %% compute C6 coefficients and Rydberg blockade radius

# C6 coefficients
c6_overn11_list = []
c6_list = []

for n in n_values_plot:
    # n has to be int for ARC
    n = int(n)
    
    # get C6 coefficients
    c6 = calculate_c6_coefficients(n, 0, 1, 0)
    c6_list.append(c6)
    
    # divide by n^11 with intermediate step to avoid integer overlow
    # where the number is too large to store in normal int64 64 bit number
    n_float = float(n)
    n_term = n_float**11
    c6_coefficient_overn11 = c6 / n_term
    
    # store result, with and without divide by n^11
    c6_overn11_list.append(c6_coefficient_overn11)

c6_overn11_array = np.array(c6_overn11_list)
c6_array = np.array(c6_list)

# Blockade radius

# C6 coeff in terms of GHz, convert to Hz
c6_Hz = c6_array[:,0] * 1e9
blockade_radii = (abs(c6_Hz) / rabi_freqs)**(1/6)

# %% plotting

matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

fig, ax = plt.subplots()
ax.grid()

ax.scatter(n_values_plot, c6_overn11_array[:,0])

ax.set_xlabel('$n$')
ax.set_ylabel('$C_6 / n^{11}$ coefficients [atomic units]')

fig2, ax2 = plt.subplots()

ax2.grid()
ax2.plot(n_values_plot, abs(c6_array[:,0]))
# ax2.plot(n_array, c6_fit_result)

ax2.set_yscale('log')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$|C_6|$ [GHz $\mu$m$^6$]')

fig3, ax3 = plt.subplots()
ax3.grid()

ax3.plot(n_values_plot, blockade_radii)

ax3.set_xlabel('$n$')
ax3.set_ylabel('Blockade radius [$\mu$m]')


plt.show()

