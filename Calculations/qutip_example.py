# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:45:49 2022

@author: s163673
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# coherent state N, alpha
x = coherent(5, 0.5-0.5j)
print(x)
result = x.check_herm()
