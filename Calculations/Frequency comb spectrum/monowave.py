# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:37:56 2022

@author: Marijn L. Venderbosch
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft

#%% variables

carrier_amplitude = 1
carrier_frequency = 2e6 # 2 MHz
modulator_frequency = 20e3 # 20 kHz
modulation_amplitude = 25 * modulator_frequency

time_step = 1e-6
data_points = int(1e6)
time = np.linspace(0, data_points * time_step, data_points)

#%% functions

def frequency_modulation_signal(time):
    carrier = 2 * np.pi * carrier_frequency * time
    modulator = modulation_amplitude / modulator_frequency * np.sin(2 * np.pi * modulator_frequency)
    signal = carrier_amplitude * np.cos(carrier + modulator)
    return signal

y = frequency_modulation_signal(time)
Y = rfft(y)
x_f = np.linspace(0, 1.0 / (2.0 * time_step), int(data_points // 2))

fig, ax = plt.subplots()
ax.stem(x_f, 2.0 / data_points * np.abs(Y[:data_points // 2]),
         markerfmt = " ",
         basefmt = "-b")
ax.set_xlim(0, 20)




