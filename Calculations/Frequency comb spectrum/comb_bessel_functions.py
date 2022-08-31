# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
It does this by computing Bessel functions of the first kind
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

#%% variables

modulation_index = 75
extra_indices = 5 # extend plotting windows beyond just modulation index
modulation_frequency = 20e3

#%% computation bessel functions

# initialize empty matrix that serves to store bessel function as function of comb line
comb_line_matrix = np.linspace(- modulation_index - extra_indices, 
                      modulation_index + extra_indices,
                      2 * modulation_index + 2 * extra_indices + 1) 

def bessel_function_firstkind(order, index):
    return jv(order, index)

amplitude_array = bessel_function_firstkind(comb_line_matrix, modulation_index)
intensity_array = amplitude_array**2

def statistics(array):
    mean = np.mean(array)
    stddev = np.std(array)
    return mean, stddev

mean_intensity, spread_intensity = statistics(intensity_array)
min_intensity = np.min(intensity_array)
max_intensity = np.max(intensity_array)

#%% plotting

fig, ax = plt.subplots()
ax.stem(comb_line_matrix, abs(amplitude_array),
        markerfmt = " ", 
        basefmt = "b")
ax.set_xlabel('comb line')
ax.set_ylabel('amplitude [a.u.]')

frequency_axis = comb_line_matrix * modulation_frequency / 1e6 #convert Hz to MHz

fig, ax = plt.subplots()
ax.stem(frequency_axis, intensity_array, 
        markerfmt = " ",
        basefmt = "b")
ax.set_xlabel('frequency offset [MHz]')
ax.set_ylabel('relative intensity [a.u.]')