# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 11:43:22 2022

@author: Marijn L. Venderbosch
"""

#%% imports

# python libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
from scipy import signal as sg
from functools import partial
from scipy.integrate import quad
from scipy.integrate import simpson

#%% variables

triangle_amplitude = 1
triangle_frequency = 1
triangle_offset = 0

carrier_frequency = 10
modulation_amplitude = 1

time_start = 0
time_stop = 10
time_steps = int(1e2) + 1
time = np.linspace(time_start, time_stop, time_steps)

#%% functions

def trianglar_waveform(amplitude, frequency, offset, input_vector):
    """
    Ouputs signal that oscillates between min. and max. values with constant slope.
    Creating a triangular shape
    """
    sawtooth_signal = sg.sawtooth(2 * np.pi * frequency * input_vector, 
                                  width = 0.5)
    signal = amplitude * sawtooth_signal + offset
    return signal

def wave_form(input_vector):
    """
    Similar to triangular_waveform function, but with variables substituted. 
    """
    signal = trianglar_waveform(triangle_amplitude,
                                triangle_frequency, 
                                triangle_offset,
                                input_vector)
    return signal

# integrating function
def indefinite_integral(function, start_point, independent_variable):
    """"
    Using combination of map() and partial() functions
    https://stackoverflow.com/questions/61675014/integral-with-variable-limits-in-python
    """
    result= np.array(
        list(map(partial(quad, function, start_point), independent_variable))
        )[:, 0]
    return result

Y = indefinite_integral(wave_form, 0, time)

#%% plotting
plt.plot(time,Y)

