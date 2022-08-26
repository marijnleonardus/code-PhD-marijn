# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:32:50 2022

@author: Marijn L. Venderbosch

Script plots triangular wave 
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg
import scipy.integrate as integrate
import scipy.special as special

#%% variables
triangle_amplitude = 1
triangle_frequency = 1
triangle_offset = 0

#%% functions
def triangular_waveform(amplitude, frequency, offset, input_vector):
    """
    Ouputs signal that oscillates between min. and max. values with constant slope.
    Creating a triangular shape
    """
    sawtooth_signal = sg.sawtooth(2 * np.pi * frequency * input_vector, width = 0.5)
    signal = amplitude * sawtooth_signal + offset
    return signal

#%% manipulation
time = np.linspace(0, 10, 1000)

y = triangular_waveform(amplitude = triangle_amplitude,
                        frequency = triangle_frequency,
                        offset = triangle_offset,
                        input_vector = time)

#%% plotting
plt.plot(time, y)
