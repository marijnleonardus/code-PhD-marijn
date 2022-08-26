# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:45:31 2022

@author: Marijn L. Venderbosch

script plots FFT of real-valued time signal
"""

#%% imports

from scipy.fft import rfft
import matplotlib.pyplot as plt
import numpy as np

#%% variables

data_points = int(1e3)
time_step = 1.0 / data_points
time = np.linspace(0.0, data_points * time_step, data_points)

#%% manipulation

# define function 
y = np.sin(60.0 * 2.0 * np.pi * time) + 0.5 * np.sin(90.0 * 2.0 * np.pi * time)

# compute FFT with real valued imput. Therefore negative frequencies are omitted. 
# This only works for real input
y_f = rfft(y)

# Fourier transformed x coordinate. Only half the data points of the time array remain
x_f = np.linspace(0.0, 1.0 / (2.0 * time_step), data_points // 2)

#%% plotting

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

ax1.plot(time, y)
ax1.set_xlabel(r'time $t$')
ax1.set_ylabel(r'$y(t)$')

ax2.stem(x_f, 2.0 / data_points * np.abs(y_f[:data_points // 2]),
         markerfmt = " ",
         basefmt = "-b")
ax2.set_xlabel('frequency $f$')
ax2.set_ylabel(r'FFT($y)$, $f>0$')
ax2.set_xlim(0, 100)

fig.tight_layout()
