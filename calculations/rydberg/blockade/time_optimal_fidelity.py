# author: Marijn Venderbosch
# 2023

import numpy as np

pi = np.pi

rabi = 2*pi*15e6
V = 20*rabi
rydberg_lifetime = 96.5e-6  
decay_rate = 1/rydberg_lifetime
Tomegamax = 7.612
alpha = 35.9

error = decay_rate*(Tomegamax)/rabi+rabi**2*alpha/(V**2*Tomegamax**2)

