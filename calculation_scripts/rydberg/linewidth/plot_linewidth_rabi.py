# author: Marijn Venderbosch
# january 2023

import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np

# user defined
from functions.conversion_functions import (wavelength_to_freq, 
                                            rdme_to_rate,
                                            gaussian_beam_intensity,
                                            cylindrical_gaussian_beam,
                                            rdme_to_rabi)

# %% variables

rdme_values = genfromtxt('calculations/rydberg/data/rdme_values.csv', delimiter=',')
rdme_values_fit = genfromtxt('calculations/rydberg/data/rdme_values_fit.csv', delimiter=',')
n_values = genfromtxt('calculations/rydberg/data/n_values.csv', delimiter = ',')
n_values_plot = genfromtxt('calculations/rydberg/data/n_values_plot.csv', delimiter=',')

# paramters Madjarov
waist = 20e-6  # m
laser_power = 20e-3  # m

# producing rabi freq vs power plot
power_array = np.linspace(0.001, 1, 100)  # W
rabi_plot_waist = 200e-6  # m

# cylindrical beam
waist_y = 20e-6  # m


# %% Compute linewidths and rabi frequencies 

# intensity
intensity = gaussian_beam_intensity(waist, laser_power)

# convert to einstein coefficients
# transition frequency 
omega21 = wavelength_to_freq(317e-9)

# einstein coefficients data points and fit
einstein_coefficients = rdme_to_rate(rdme_values, 0, omega21, 0)
einstein_coefficients_fit = rdme_to_rate(rdme_values_fit, 0, omega21, 0)

# compute rabi frequency as a function of n for madajarov parameters
rabi_freqs = rdme_to_rabi(rdme_values, intensity)
rabi_freqs_fit = rdme_to_rabi(rdme_values_fit, intensity)

# compute rabi frequency as a function of n for our parameters and save result
intensity_ourparameters = cylindrical_gaussian_beam(rabi_plot_waist, waist_y, laser_power)
rabi_fres_ourparam = rdme_to_rabi(rdme_values_fit, intensity_ourparameters)
np.savetxt('calculations/Rydberg/data/rabi_freq_vs_n.csv', rabi_fres_ourparam, delimiter=',')

# compute rabi freq as a function of power for n=61
n_61 = np.where(n_values_plot==61)
rdme_61 = rdme_values_fit[n_61]

# compute gaussian beam rabi frequencies
rabi_array = []
cylindrical_rabi_array = []

for power in power_array:
    # circular gaussian beam
    intensity_gaussian = gaussian_beam_intensity(rabi_plot_waist, power)
    rabi = rdme_to_rabi(rdme_61, intensity_gaussian)
    rabi_array.append(rabi)
    
    # cylindrical gaussian beam
    intensity_cylindrical = cylindrical_gaussian_beam(rabi_plot_waist, waist_y, power)
    rabi_cylindrical = rdme_to_rabi(rdme_61, intensity_cylindrical)
    cylindrical_rabi_array.append(rabi_cylindrical)
    
# convert to np array
rabi_array = np.array(rabi_array)
cylindrical_rabi_array = np.array(cylindrical_rabi_array)

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

# Rabi frequency vs n
fig3, ax3 = plt.subplots()
ax3.grid()

ax3.plot(n_values_plot, rabi_freqs_fit / (2*np.pi) / 1e6, # convert to 2 pi * MHz
         label='$P=20$ mW, $w_0 = 20$ $\mu$m')

ax3.set_xlabel('$n$')
ax3.set_ylabel(r'Rabi frequency $\Omega$ [$2 \pi \cdot$ MHz]')
ax3.legend()

# beam power against Rabi freq and scattering rate
fig4, ax4 = plt.subplots()
ax4.grid()

ax4.plot(power_array / 1e-3, # convert to mW
         rabi_array / (2*np.pi) / 1e6,
         'b-',
         label=f'$n=61$, $w_0={rabi_plot_waist*1e6}$ $\mu$m')
ax4.plot(power_array / 1e-3,
         cylindrical_rabi_array / (2*np.pi) / 1e6,
         'b--',
         label=f'$n=61$, $w_x={rabi_plot_waist*1e6}$ $\mu$m, $w_y={waist_y*1e6}$ $\mu$m')

ax4.set_xlabel('Laser power [mW]')
ax4.set_ylabel(r'Rabi frequency $\Omega$ [$2\pi \cdot$ MHz]')
ax4.legend()
