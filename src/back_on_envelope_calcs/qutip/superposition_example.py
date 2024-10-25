# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:00:07 2023

@author: s163673
"""

from qutip import *

ket=(basis(5,0)+basis(5,1)).unit()
print(ket)

n = num(5)

print(n*ket)