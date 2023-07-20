# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
it does this by computing the modulated laser in the time domain and computing FFT
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from freq_mod_class import FrequencyModulation

# %% variables

# broadband red MOT parameters
mod_depth = 2*1e6  # [Hz]
mod_freq = 50*1e3  # [1/s]
carrier_freq = 80*1e6  # [1/s] 

# fft parameters
sample_freq = int(carrier_freq*100)  # [1/s], s
time_window = 10*1/(mod_freq)
time_array = np.arange(0, time_window, 1/sample_freq)  # [s]
nyquist_freq = sample_freq//2  # [1/s]

# %% main

def main():
    # Compute frequency modulated signal from linear ramp
    fm_signal = FrequencyModulation(mod_freq, mod_depth, carrier_freq).linear_frequency_ramp(time_array)

    # Compute Fourier transform
    freq_domain_signal = np.fft.fft(fm_signal)

    # Compute correct frequency axes for plotting
    nr_samples = len(freq_domain_signal)
    sampling_period = nr_samples / sample_freq
    freqs = np.arange(nr_samples) / sampling_period

    # Remove second half, only signal up to Nyquist freq. is relevant
    freqs = freqs[:nr_samples // 2]
    freq_domain_signal = freq_domain_signal[:nr_samples // 2]

    # Plot frequency domain signal
    plt.plot(freqs, np.abs(freq_domain_signal))
    plt.xlim(carrier_freq - 1.5 * mod_depth, carrier_freq + 0.5 * mod_depth)
    plt.loglog()
    plt.show()

if __name__=="__main__":
    main()

# %%
