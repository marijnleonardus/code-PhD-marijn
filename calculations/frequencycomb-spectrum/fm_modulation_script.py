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

sample_freq=int(10e7) # 1/s
nyquist_freq=sample_freq/2 # 1/s

time_window=1e-3  # s
t=np.arange(0, time_window, 1/sample_freq)  # s

mod_freq=20e3 # 1/s
carrier_freq=2e6 # 1/s for computational reasons this is lower. will only shift peak x coordinate
mod_index=75

nr_samples=int(sample_freq*time_window)
T=time_window/nr_samples


#%% modulate FM spectruma and plot FFT

def FM_modulation(t):
    phi = 2 * np.pi * carrier_freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t)
    FM = np.cos(phi)
    return FM

Y = fft(FM_modulation(t))
xf = fftfreq(nr_samples, T)[:nr_samples//2]


#%% plotting

fig, ax = plt.subplots()
ax.stem(xf/1e6, 2.0/nr_samples*np.abs(Y[0:nr_samples//2]), markerfmt = " ", basefmt = "b")
ax.set_xlim((carrier_freq-1.5*mod_index*mod_freq)/1e6, (carrier_freq+1.5*mod_index*mod_freq)/1e6)
ax.set_xlabel('frequency [MHz]')
ax.set_ylabel('amplitude [a.u.]')
ax.set_title('amplitude')

fig, ax = plt.subplots()
ax.stem(xf/1e6, 2.0/nr_samples*np.abs(Y[0:nr_samples//2])**2, markerfmt=" ", basefmt="b")
ax.set_xlim((carrier_freq-1.5*mod_index*mod_freq)/1e6, (carrier_freq+1.5*mod_index*mod_freq)/1e6)
ax.set_xlabel('frequency [MHz]')
ax.set_ylabel('intensity [a.u.]')
ax.set_title('intensity')
