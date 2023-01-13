# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
it does this by computing the modulated laser in the time domain and computing FFT
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq

#%% variables

sampling_frequency = int(10e7) # 1/s
nyquist_frequency = sampling_frequency / 2 # 1/s

time_window = 1e-3 # s
t = np.arange(0, time_window, 1 / sampling_frequency) # s

modulation_frequency = 20e3 # 1/s
carrier_frequency = 2e6 # 1/s for computational reasons this is lower. will only shift peak x coordinate
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

#%% plotting

fig, ax = plt.subplots()
ax.stem(xf, 2.0 / number_sampling_points * np.abs(Y[0 : number_sampling_points // 2]), 
        markerfmt = " ",
        basefmt = "b")

ax.set_xlim(carrier_frequency - 1.5 * modulation_index * modulation_frequency, 
            carrier_frequency + 1.5 * modulation_index * modulation_frequency)
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('amplitude [a.u.]')
ax.set_title('amplitude')

fig, ax = plt.subplots()
ax.stem(xf, 2.0 / number_sampling_points * np.abs(Y[0 : number_sampling_points // 2])**2, 
        markerfmt = " ",
        basefmt = "b")

ax.set_xlim(carrier_frequency - 1.5 * modulation_index * modulation_frequency, 
            carrier_frequency + 1.5 * modulation_index * modulation_frequency)
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('intensity [a.u.]')
ax.set_title('intensity')