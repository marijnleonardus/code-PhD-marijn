# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
it does this by computing the modulated laser in the time domain and computing FFT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq
from scipy import signal
from scipy.integrate import quad

#%% variables

pi = np.pi

sample_freq = int(10e7) # 1/s
nyquist_freq = sample_freq/2 # 1/s

time_window = 1e-3  # s
time_array = np.arange(0, time_window, 1/sample_freq)  # s

mod_freq = 20e3 # 1/s
carrier_freq = 10e6 # 1/s for computational reasons this is lower. will only shift peak x coordinate
mod_index = 75

nr_samples = int(sample_freq*time_window)
carrier_period = time_window/nr_samples


#%% modulate FM spectruma and plot FFT

def FM_modulation_sinuoidal(time_array):
    phi = 2 * np.pi * carrier_freq * time_array + mod_index * np.sin(2 * np.pi * mod_freq * time_array)
    FM = np.cos(phi)
    return FM


def linear_ramp(fmod, freq_carrier, index_mod, time_array):
    """
    linear frequency modulation ramp, to write to the DDS

    Parameters
    ----------
    fmod : float
        modulation freq = 1/T [Hz].
    freq_carrier : float
        carrier freq: AOM freq in [Hz].
    index_mod : int
        max. deviation of carrier freq = index_mod*fmod [Hz].

    Returns
    -------
    ramp : array of float
        the ramp to write to the DDS.

    """
    sawtooth = -0.5*signal.sawtooth(2*pi*fmod*time_array) - 0.5
    ramp = freq_carrier + index_mod*fmod*sawtooth
    return ramp


ramp = linear_ramp(mod_freq, carrier_freq, mod_index, time_array)


FM = np.zeros(len(time_array))

for t in range(len(time_array)):
    phase = quad(linear_ramp(mod_freq, carrier_freq, mod_index), 0, t)
    d = np.cos(phase)

Y = fft(FM_modulation_sinuoidal(time_array))
xf = fftfreq(nr_samples, carrier_period)[:nr_samples//2]


#%% plotting

fig, ax = plt.subplots()
ax.plot(time_array, ramp)
ax.set_xlim(0, 5*1/mod_freq)
ax.set_xlabel('time')
ax.set_ylabel('frequency')

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
