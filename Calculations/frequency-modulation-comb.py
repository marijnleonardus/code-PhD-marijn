#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on August 8 2022
@author: Marijn L. Venderbosch

plots FFT (frequency response) of frequency modulation
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

#%% variables
modulator_frequency = 20e3 # Hz
carrier_frequency = 200e6 # Hz
modulation_index = 3
time_interval = 1e-4 # seconds
sampling_rate = 1e-7

#%% manipulation

# compute frequency modulation signal
time = np.arange(0, time_interval, sampling_rate)
modulator = np.sin(2.0 * np.pi * modulator_frequency * time) * modulation_index
carrier = np.sin(2.0 * np.pi * carrier_frequency * time)
product = np.zeros_like(modulator)

for i, t in enumerate(time):
    product[i] = np.sin(2. * np.pi * (carrier_frequency * t + modulator[i]))

# compute FFT
fourier = fft(product)
data_points = len(fourier)
point = np.arange(data_points)
time = data_points / sampling_rate
frequency = point / time 
    
#%% plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 1, nrows = 4,
                       figsize = (5, 11))
ax1.plot(modulator)
ax1.set_ylabel('amplitude')

ax2.plot(carrier)
ax2.set_ylabel('amplitude')

ax3.plot(product)
ax3.set_ylabel('amplitude')

ax4.plot(frequency, np.abs(fourier))
