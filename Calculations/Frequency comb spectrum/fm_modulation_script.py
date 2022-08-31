# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq
from scipy.special import jv

#%% variables

sampling_frequency = int(10e7) # 1/s
nyquist_frequency = sampling_frequency / 2 # 1/s

time_window = 1e-3 # s
t = np.arange(0, time_window, 1 / sampling_frequency) # s

modulation_frequency = 20e3 # 1/s
carrier_frequency = 2e6 # 1/s
modulation_index = 75

number_sampling_points = int(sampling_frequency * time_window)
T = time_window / number_sampling_points

#%% modulate FM spectruma and plot FFT

def FM_modulation(t):
    phi = 2 * np.pi * carrier_frequency * t + modulation_index * np.sin(2 * np.pi * modulation_frequency * t)
    FM = np.cos(phi)
    return FM

Y = fft(FM_modulation(t))
xf = fftfreq(number_sampling_points, T)[:number_sampling_points // 2]

def bessel_function_firstkind(order, index):
    return jv(order, index)

# initialize empty matrices
indices = np.linspace(- modulation_index -5, 
                      modulation_index + 5,
                      2 * modulation_index + 11) 

amplitude_array = bessel_function_firstkind(indices, modulation_index)

fig, ax = plt.subplots()
ax.stem(indices, abs(amplitude_array), markerfmt = " ", basefmt = "b")
ax.set_xlabel('comb line')
ax.set_ylabel('amplitude [a.u.]')

#%% plotting

fig, ax = plt.subplots()
ax.stem(xf, 2.0 / number_sampling_points * np.abs(Y[0 : number_sampling_points // 2]), 
        markerfmt = " ",
        basefmt = "b")
ax.set_xlim(carrier_frequency - 1.5 * modulation_index * modulation_frequency, 
            carrier_frequency + 1.5 * modulation_index * modulation_frequency)

frequency_axis = indices * 20e3 / 1e6 #convert to MHz

fig, ax = plt.subplots()
ax.stem(frequency_axis, abs(amplitude_array)**2, 
        markerfmt = " ",
        basefmt = "b")
ax.set_xlabel('frequency offset [MHz]')
ax.set_ylabel('relative intensity [a.u.]')