# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 10:50:27 2022

@author: Marijn L. Venderbosch

example showing how to compute indefinite integral numerically
"""

#%% imports
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from functools import partial

#%% manipulation
# function to be integrated
def y(x):
    return x**2

# integrating function
def Y(x):
    """"
    Using combination of map() and partial() functions
    https://stackoverflow.com/questions/61675014/integral-with-variable-limits-in-python
    """
    result= np.array(
        list(map(partial(quad, y, 0), x))
        )[:, 0]
    return result

#%% plotting
x = np.linspace(0, 3, 100)
fig, ax = plt.subplots()
ax.plot(x, Y(x))

# compare to exact result with offset of 0.1 to avoid overlapping lines
ax.plot(x, 1/3 * x**3 + 0.1)

