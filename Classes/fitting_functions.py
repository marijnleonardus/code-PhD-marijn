# author: Marijn Venderbosch
# january 2022

import numpy as np


def fit_func_rdme(x, a1, a2, b1, b2):
    # function for fitting exp. data
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2*x**2)