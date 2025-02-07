# author: Marijn Venderbosch
# January 2025

import sys
import os
from scipy.constants import physical_constants, epsilon_0, c, pi
bohr_magneton = physical_constants['Bohr magneton'][0]
import numpy as np
import matplotlib.pyplot as plt

# append path with 'modules' dir in parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries 
from optics_class import GaussianBeam

# variables
tweezer_power = 90*1e-3/4 # W
tweezer_waist = 0.8e-6 # m
vector_pol = 1
gj = 1
j = 1
wavelength = 813e-9 # m

Tweezer = GaussianBeam(tweezer_power, tweezer_waist)
max_intensity = Tweezer.get_intensity()
theta = wavelength/pi/tweezer_waist

# create 2d array
x = np.linspace(-1e-6, 1e-6, 100)
y = np.linspace(-1e-6, 1e-6, 100)
x, y = np.meshgrid(x, y)

fictional_b_field = -vector_pol/(bohr_magneton*gj*j)*(-2*theta*y/tweezer_waist*2*max_intensity**2/(epsilon_0*c)*np.exp(-2*(x**2+y**2)/tweezer_waist**2))

fig, ax = plt.subplots()
ax.imshow(fictional_b_field, cmap='bwr', interpolation='nearest')
cax = ax.imshow(fictional_b_field, cmap='bwr', interpolation='nearest', extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
fig.colorbar(cax, ax=ax)  # Add colorbar to the figure
ax.set_xlabel(r'x [$\mu$m]')
ax.set_ylabel(r'y [$\mu$m]')
plt.show()