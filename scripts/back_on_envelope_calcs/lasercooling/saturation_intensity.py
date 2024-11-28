import numpy as np
from scipy.constants import pi
import matplotlib.pyplot as plt
import sys
import os

# import module from '../modules'
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)
from conversion_class import Conversion
from optics_class import GaussianBeam

# variables
linewidth = 2*pi*7.4e3  # Hz
lifetime = 1/linewidth  # s
wavelength = 689e-9   # m
beam_power_sisyphus = 0.1*1e-3  # W
fiber_na = 0.08
f_collimationlens = 30e-3  # m
beam_waist_sisyphus = fiber_na*f_collimationlens  # m

# compute saturation intensity
saturation_intensity = Conversion.saturation_intensity(lifetime, wavelength)  # W/m^2
SisyphusLaser = GaussianBeam(beam_power_sisyphus, beam_waist_sisyphus)
saturation_param = SisyphusLaser.get_intensity()/saturation_intensity
print(saturation_param)
