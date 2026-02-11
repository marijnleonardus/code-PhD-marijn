import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import proton_mass, pi, hbar
import sys
import os

# Import custom modules
from modules.optics_class import GaussianBeam
from modules.laser_class import AtomLightInteraction
from utils.units import um, mW, nm, MHz

plt.style.use('default')
os.system('cls' if os.name == 'nt' else 'clear')

# variables
uv_wavelength = 316.6*nm  # [m]
uv_beam_waist = 85*um  # [m], see madjarov 2020
uv_beam_power = 43*mW  # power in rydberg laser at atoms in [W], see madjarov 2020
n = 61

RydbergBeam = GaussianBeam(uv_beam_power, uv_beam_waist)
rydberg_intensity = RydbergBeam.get_intensity()
rabi_freq = AtomLightInteraction.calc_rydberg_rabi_freq(n, rydberg_intensity, j_e=1)
print(np.round(rabi_freq/(2*pi*MHz), 2), " MHz")
