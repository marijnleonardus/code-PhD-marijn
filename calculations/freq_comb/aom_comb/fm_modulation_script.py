# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:16:43 2022

@author: Marijn Venderbosch

script plots spectrum of frequency modulated laser, generating a broadband comb
it does this by computing the modulated laser in the time domain and computing FFT
"""

import numpy as np
import matplotlib.pyplot as plt
from freq_mod_class import FrequencyModulation

# broadband red MOT parameters
mod_depth = 5*1e6  # [Hz]
mod_freq = 50*1e3  # [1/s]
carrier_freq = 80*1e6  # [1/s] 

# sampling parameters
sample_overhead = 10  # how much higher sampling freq. is than carrier freq. 
nr_ramps_to_sample = 100  # nr of freq. ramps to sample in time domain

mhz = 1e6  # [Hz]

def main():

    # FFT parameters
    sample_freq = int(carrier_freq*sample_overhead)  # [1/s], 100 samples per carrier period
    time_window = nr_ramps_to_sample/mod_freq  # [s]
    time_array = np.arange(0, time_window, 1/sample_freq)  # [s]

    # Compute frequency modulated signal from linear ramp and tringular ramp
    RedMotDDS = FrequencyModulation(mod_freq, mod_depth, carrier_freq)
    t_signal_linear = RedMotDDS.linear_frequency_ramp(time_array)
    t_signal_triangular = RedMotDDS.triangular_frequency_modulation(time_array)

    # Compute Fourier transform
    fft_signal = np.fft.fft(t_signal_triangular)

    # Compute correct frequency axes for plotting
    nr_samples = len(fft_signal)
    sampling_period = nr_samples / sample_freq
    freqs = np.arange(nr_samples) / sampling_period

    # Remove second half, only signal up to Nyquist freq. is relevant
    freqs = freqs[:nr_samples // 2]
    fft_signal = fft_signal[:nr_samples // 2]

    # Normalize y axis
    fft_signal = fft_signal / np.max(np.abs(fft_signal))

    # Plot frequency domain signal
    fig, ax = plt.subplots()
    ax.plot(freqs/1e6, np.abs(fft_signal), label='FM signal')
    
    # Set axes limits, labels
    lower_xlim = (carrier_freq - 1.5 * mod_depth*.5)/mhz
    upper_xlim = (carrier_freq + 0.5 * mod_depth*.5)/mhz
    ax.set_xlim(lower_xlim, upper_xlim)
    ax.set_ylim(0, 1.05*np.max(np.abs(fft_signal)))
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Intensity [a.u.]")

    # plot carrier freq. only for comparison
    ax.axvline(x=carrier_freq/mhz, color="red", ls='--', label='carrier freq.')

    ax.legend()
    plt.show()

if __name__=="__main__":
    main()
