# author: Marijn Venderbosch
# 2023

"""estimates cavity roll off factor
for our FC1500 quantum ORS
"""

import numpy as np
import matplotlib.pyplot as plt

cavity_finesse = 315e3
cavity_linewidth = 20e3

def cavity_rolloff_factor(F, nu, f):
    """
    Parameters
    ----------
    F : float
        cavity finesse.
    nu : float
        cavity linewidth in [Hz].
    f : np array or float
        frequency in [Hz]. independent variable to compute rolloff as a function of.

    Returns
    -------
    rolloff : float
        cavity roll of factor.

    """
    rolloff = 1/F + (1+1/F)/(1+4*f**2/nu**2)
    return rolloff

frequencies = np.logspace(3, 7, num=100, base=10)
rollofffactors = cavity_rolloff_factor(cavity_finesse, cavity_linewidth, frequencies)

plt.plot(frequencies, rollofffactors)