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
from conversion_class import Conversion

atomic_unit_pol = Conversion.get_atomic_pol_unit()

# variables
tweezer_power = 200*1e-3/4 # W
waist = 0.8e-6 # m
vector_pol_3p1 = -1.22*atomic_unit_pol
gj = 3/2
J = 1
wavelength = 813e-9 # m

Tweezer = GaussianBeam(tweezer_power, waist)
max_intensity = Tweezer.get_intensity()
theta = wavelength/(pi*waist)
e_field = np.sqrt(2*max_intensity/(epsilon_0*c))

# create 2d array
x = np.linspace(-1e-6, 1e-6, 100)
y = np.linspace(-1e-6, 1e-6, 100)
x, y = np.meshgrid(x, y)

# calculate fict magn field from 10.1088/1674-1056/ad84d0
fictional_b_field_tesla = -vector_pol_3p1/(bohr_magneton*gj*J)*(-2*theta*y/waist*e_field**2*np.exp(-2*(x**2+y**2)/waist**2))
fictional_b_field_gauss = fictional_b_field_tesla*1e4

fig, ax = plt.subplots()
ax.imshow(fictional_b_field_gauss, cmap='bwr', interpolation='nearest')
cax = ax.imshow(fictional_b_field_gauss, cmap='bwr', interpolation='nearest', extent=[x.min()*1e6, x.max()*1e6, y.min()*1e6, y.max()*1e6])
fig.colorbar(cax, ax=ax)  # Add colorbar to the figure
ax.set_xlabel(r'x [$\mu$m]')
ax.set_ylabel(r'y [$\mu$m]')
ax.set_title(r'Fictitious magnetic field [G]')
plt.show()