# author: Marijn Venderbosch
# january 2023

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

# user defined
from functions.conversion_functions import wavelength_to_freq, rdme_to_rate


# %% variables

rdme_values = genfromtxt('calculations/rydberg/data/rdme_values.csv', delimiter=',')
rdme_values_fit = genfromtxt('calculations/rydberg/data/rdme_values_fit.csv', delimiter=',')
n_values = genfromtxt('calculations/rydberg/data/n_values.csv', delimiter = ',')
n_values_plot = genfromtxt('calculations/rydberg/data/n_values_plot.csv', delimiter=',')

# %%% manipulation

# convert to einstein coefficients
# transition frequency 
omega21 = wavelength_to_freq(317e-9)

# einstein coefficients data points
einstein_coefficients = rdme_to_rate(rdme_values, 0, omega21, 0)

# einstein coefficients fit
einstein_coefficients_fit = rdme_to_rate(rdme_values_fit, 0, omega21, 0)

# %% Plotting

# RDME values
fig, ax = plt.subplots()

ax.scatter(n_values, rdme_values,
           label='data')
ax.plot(n_values_plot, rdme_values_fit, 'r--', label='fit')

ax.grid()
ax.set_xlim(18, 70)
ax.set_xlabel('$n$')
ax.set_ylabel('RDME [atomic units]')
ax.legend()

# einstein coefficients
fig2, ax2 = plt.subplots()

ax2.grid()
ax2.set_xlabel('$n$')
ax2.set_ylabel('Einstein coefficient [$2\pi \cdot Hz$]')
ax2.scatter(n_values, einstein_coefficients, label='data')
ax2.plot(n_values_plot, einstein_coefficients_fit, 'r--', label='fit')
ax2.legend()

# insert zoom
# axins = zoomed_inset_axes(ax2, 3, loc='upper right', 
#                           axes_kwargs={"facecolor" : "lightgray"})

# axins.plot(n_values_plot, einstein_coefficients_fit /2 / np.pi)
# axins.set_xlim(55, 65)
# axins.set_ylim(0, 500)