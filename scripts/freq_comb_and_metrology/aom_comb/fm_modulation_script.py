# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
it does this by computing the modulated laser in the time domain and computing FFT
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy import signal
from scipy.integrate import cumulative_trapezoid
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
utils_dir = os.path.abspath(os.path.join(script_dir, '../../../utils'))
sys.path.append(modules_dir)
sys.path.append(utils_dir)

from units import MHz, kHz
from plot_utils import Plotting

# broadband red MOT parameters
mod_depth = 2*1.7*MHz  # [Hz]
mod_freq = 49*kHz  # [1/s]
aom_center_freq = 10*MHz  # [1/s]
detuning = -450*kHz  # [1/s]
carrier_freq = aom_center_freq + detuning

# sampling parameters
sample_overhead = 10  # how much higher sampling freq. is than carrier freq. 
nr_ramps_to_sample = 100  # nr of freq. ramps to sample in time domain

# FFT parameters
sample_freq = int(carrier_freq*sample_overhead)  # [1/s], 100 samples per carrier period
time_window = nr_ramps_to_sample/mod_freq  # [s]
time_array = np.arange(0, time_window, 1/sample_freq)  # [s]

def triangular_frequency_modulation(t: np.ndarray, mod_freq: float, mod_depth: float, carrier_freq: float) -> np.ndarray:
    """computes frequency modulated signal for triangular frequency modulation

    Args:
        t (np.ndarray): time, matrix

    Returns:
        np.ndarray: frequency as a function of time
    """

    # 0.5 means start going back halfway, producing triangular waveform
    mod_signal = -0.25*(1 + signal.sawtooth(2*pi*mod_freq*t, 0.5))

    # frequency modulation, formula from wiki page
    # because 'cumtrapz' function does not work on first entry, 
    # remove first entry of 't' matrix
    integral = cumulative_trapezoid(mod_signal, t)
    phase = 2*pi*carrier_freq*t[1::] + 2*pi*mod_depth*integral
    fm_signal = np.cos(phase)
    return fm_signal


# Compute frequency modulated signal from linear ramp and tringular ramp
t_signal_triangular = triangular_frequency_modulation(time_array, mod_freq, mod_depth, carrier_freq)

# Compute Fourier transform
fft_signal = np.fft.fft(t_signal_triangular)

# Compute correct frequency axes for plotting
nr_samples = len(fft_signal)
sampling_period = nr_samples/sample_freq
freqs = np.arange(nr_samples)/sampling_period

# Remove second half, only signal up to Nyquist freq. is relevant
freqs = freqs[:nr_samples // 2]
fft_signal = fft_signal[:nr_samples // 2]

# Normalize y axis
fft_signal = fft_signal/np.sum(np.abs(fft_signal))

fig, ax = plt.subplots(figsize=(3.8, 2.3))
ax.plot((freqs - carrier_freq)/MHz, np.abs(fft_signal))
lower_xlim = (carrier_freq - 1.5*mod_depth*.5)/MHz - carrier_freq/MHz
upper_xlim = (carrier_freq + 0.5*mod_depth*.5)/MHz - carrier_freq/MHz
ax.set_xlim(lower_xlim, upper_xlim)
ax.set_ylim(0, 1.05*np.max(np.abs(fft_signal)))
ax.set_xlabel("Detuning [MHz]")
ax.set_ylabel("Intensity [a.u.]")

Plot = Plotting('output')
Plot.savefig('fm_modulation_spectrum.png')
Plot.savepgf('fm_mod')
