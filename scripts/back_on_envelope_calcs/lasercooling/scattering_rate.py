import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
import sys
import os

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../../modules', '../../../utils'])

from units import MHz, mW, mm, nm
from laser_class import AtomLightInteraction
from optics_class import GaussianBeam

wavelength = 461e-9  # m
gamma = 2*pi*32*MHz  # Hz
det_deflector = -30*MHz  # Hz
power_deflector = 30*mW # W
waist_deflector = 4.8*mm  # m

DeflectorBeam = GaussianBeam(power_deflector, waist_deflector)
intensity_deflector = DeflectorBeam.get_intensity()
scattering_rate = AtomLightInteraction().scattering_rate_sat(intensity_deflector, det_deflector, gamma, 461*nm)
print('scattering rate deflector beam: ' + str(scattering_rate/MHz) + ' MHz')
