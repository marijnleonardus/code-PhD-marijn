# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
It does this by 
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

#%% variables

modulation_index = 75
extra_indices = 5 # extend plotting windows beyond just modulation index
modulation_frequency = 20e3

#%% modulate FM spectruma and plot FFT

# initialize empty matrices
indices = np.linspace(- modulation_index - extra_indices, 
                      modulation_index + extra_indices,
                      2 * modulation_index + 2 * extra_indices + 1) 

def bessel_function_firstkind(order, index):
    return jv(order, index)

amplitude_array = bessel_function_firstkind(indices, modulation_index)

#%% plotting

fig, ax = plt.subplots()
ax.stem(indices, abs(amplitude_array),
        markerfmt = " ", 
        basefmt = "b")
ax.set_xlabel('comb line')
ax.set_ylabel('amplitude [a.u.]')

frequency_axis = indices * modulation_frequency / 1e6 #convert Hz to MHz

fig, ax = plt.subplots()
ax.stem(frequency_axis, abs(amplitude_array)**2, 
        markerfmt = " ",
        basefmt = "b")
ax.set_xlabel('frequency offset [MHz]')
ax.set_ylabel('relative intensity [a.u.]')