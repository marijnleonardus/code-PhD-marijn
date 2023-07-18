# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:45:49 2022

@author: s163673
"""

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

vac = basis(5, 0)
print(vac)

a = destroy(5)
print(a)

print(a*vac)

c = create(5)
print(c *c* vac)