import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
import sys
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'modules' directory
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))

# Add the 'modules' directory to the Python path
sys.path.append(modules_dir)

from rates_class import LightAtomInteraction


gamma = 2*pi*32e6  # Hz
det = -30e6  # Hz
sat = np.linspace(0, .001, 100)

# formula 9.4, foot atomic physics
scattering_rate = LightAtomInteraction.scattering_rate_sat(detuning = det, linewidth=gamma, s=sat)
scattering_rate_khz = scattering_rate/1e3

plt.plot(sat, scattering_rate_khz)
plt.xlabel(r'saturation parameter $s$')
plt.ylabel(r'scattering rate $R$ (kHz)')
plt.show()