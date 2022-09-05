# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
It does this by computing Bessel functions of the first kind
"""

# %% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# %% variables

modulation_frequency = 20e3  # two pi * Hz
laser_power_689 = 10  # mW

modulation_index = 70
extra_indices = 5  # extend plotting windows beyond just modulation index

saturation_intensity = 0.0295  # W/m^2
beam_waist = 5e-3  # m

s_parameter = 4  # s = I/I_sat

# %% computation bessel functions

# initialize empty matrix that serves to store bessel function as function of comb line
comb_line_matrix = np.linspace(- modulation_index - extra_indices,
                               modulation_index + extra_indices,
                               2 * modulation_index + 2 * extra_indices + 1)


def bessel_function_firstkind(order, index):
    return jv(order, index)


amplitude_array = bessel_function_firstkind(comb_line_matrix, modulation_index)
relative_power_array = amplitude_array ** 2
power_array = laser_power_689 * relative_power_array

saturation_power_mw = saturation_intensity * np.pi * beam_waist ** 2 * 1e3


def statistics(array):
    mean = np.mean(array)
    stddev = np.std(array)
    return mean, stddev


mean_power, spread_power = statistics(power_array)
print(spread_power * 10e3)

# %% plotting

#  plot as a function of comb line
#  fig1, ax1 = plt.subplots()
#  ax1.stem(comb_line_matrix, abs(amplitude_array),
#           markerfmt=" ",
#           basefmt="b")
#  ax1.set_xlabel('comb line')
#  ax1.set_ylabel('amplitude [a.u.]')

#  plot as a function of frequency
frequency_axis = comb_line_matrix * modulation_frequency / 1e6  # convert Hz to MHz

fig2, ax2 = plt.subplots()
ax2.stem(frequency_axis, np.abs(amplitude_array),
         markerfmt=" ",
         basefmt="b")
ax2.set_xlabel('frequency offset [MHz]')
ax2.set_ylabel('relative amplitude [a.u.]')
ax2.set_title('electric field amplitude')

fig3, ax3 = plt.subplots()
ax3.stem(frequency_axis, power_array,
         markerfmt=" ",
         basefmt="b",
         label='power')
ax3.set_xlabel('frequency offset [MHz]')
ax3.set_ylabel('power [mW]')
ax3.set_title('power per comb line')
ax3.axhline(y=s_parameter * saturation_power_mw,
            color='r',
            linestyle='-',
            label='saturation power')
ax3.legend()

# %% include tranmission AOM

# Up until now we assumed 100% transmission of the fiber AOM
# Already for 2 MHz deviation there is some loss
# Model this loss as a parabola which could describe picture in G&H catalogue


def parabola(x, offset, x_0, a):
    return offset + a * (x - x_0) ** 2


insertion_loss_db = parabola(frequency_axis, 0, 0, 0.03)
transmission_linear = 10 ** -insertion_loss_db

fig4, ax4 = plt.subplots()
ax4.plot(frequency_axis, transmission_linear)
ax4.set_xlabel('frequency offset [MHz]')
ax4.set_ylabel('Normalized transmission')
ax4.set_title('tranmission fiber AOM')

fig5, ax5 = plt.subplots()
ax5.stem(frequency_axis, transmission_linear * power_array,
         markerfmt=" ",
         basefmt="b",
         label='power')
ax5.set_xlabel('frequency offset [MHz]')
ax5.set_ylabel('power [mW]')
ax5.set_title('power per comb line')
ax5.axhline(y=s_parameter * saturation_power_mw,
            color='r',
            linestyle='-',
            label='saturation power')
ax5.legend()

plt.show()
