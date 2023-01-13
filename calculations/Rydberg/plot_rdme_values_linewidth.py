# author: Marijn Venderbosch
# january 2023

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np

# user defined
from functions.conversion_functions import (wavelength_to_freq, 
                                            rdme_to_rate,
                                            gaussian_beam_intensity,
                                            rdme_to_rabi)


# %% variables

rdme_values = genfromtxt('calculations/rydberg/data/rdme_values.csv', delimiter=',')
rdme_values_fit = genfromtxt('calculations/rydberg/data/rdme_values_fit.csv', delimiter=',')
n_values = genfromtxt('calculations/rydberg/data/n_values.csv', delimiter = ',')
n_values_plot = genfromtxt('calculations/rydberg/data/n_values_plot.csv', delimiter=',')

waist = 20e-6
laser_power = 20e-3

# Computatoins 

# intensity
intensity = gaussian_beam_intensity(waist, laser_power)

# convert to einstein coefficients
# transition frequency 
omega21 = wavelength_to_freq(317e-9)

# einstein coefficients data points and fit
einstein_coefficients = rdme_to_rate(rdme_values, 0, omega21, 0)
einstein_coefficients_fit = rdme_to_rate(rdme_values_fit, 0, omega21, 0)

# compute rabi frequency
rabi_freqs = rdme_to_rabi(intensity, rdme_values)
rabi_freqs_fit = rdme_to_rabi(intensity, rdme_values_fit)

# %% Plotting

# RDME values
fig, ax = plt.subplots()
ax.grid()

ax.scatter(n_values, rdme_values,
           label='data')
ax.plot(n_values_plot, rdme_values_fit, 'r--', label='fit')


ax.set_xlim(18, 70)
ax.set_xlabel('$n$')
ax.set_ylabel('RDME [atomic units]')
ax.legend()

# einstein coefficients
fig2, ax2 = plt.subplots()
ax2.grid()

ax2.scatter(n_values, einstein_coefficients, label='data')
ax2.plot(n_values_plot, einstein_coefficients_fit, 'r--', label='fit')

ax2.set_xlabel('$n$')
ax2.set_ylabel('Einstein coefficient [$2\pi \cdot Hz$]')
ax2.legend()

# Rabi frequency 
fig3, ax3 = plt.subplots()
ax3.grid()

ax3.plot(n_values_plot, rabi_freqs_fit / (2*np.pi))

ax3.set_xlabel('$n$')
ax3.set_ylabel(r'$\Omega$ [$2 \pi \cdot$ Hz]')

# insert zoom
# axins = zoomed_inset_axes(ax2, 3, loc='upper right', 
#                           axes_kwargs={"facecolor" : "lightgray"})

# axins.plot(n_values_plot, einstein_coefficients_fit /2 / np.pi)
# axins.set_xlim(55, 65)
# axins.set_ylim(0, 500)