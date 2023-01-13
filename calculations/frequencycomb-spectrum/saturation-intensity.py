# -*- coding: utf-8 -*-
"""
Created on Mon Sep 5 2022

@author: Marijn Venderbosch

script computes saturation intensity of Sr red intercombination line
"""

import numpy as np
from scipy.constants import hbar, c

pi = np.pi
wavelength = 689e-9  # m
linewidth = 2 * pi * 7.4e3  # Hz


def saturation_intensity(gamma, lam):
    return 2 * pi ** 2 * hbar * c * gamma / (3 * lam ** 3)


I_sat = saturation_intensity(linewidth, wavelength)
