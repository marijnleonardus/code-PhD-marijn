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
beam_power_sisyphus = 50e-6  # W
beam_waist_sisyphus = 0.08*30e-3 # m

# compute saturation intensity
saturation_intensity = Conversion.saturation_intensity(lifetime, wavelength)  # W/m^2
SisyphusLaser = GaussianBeam(beam_power_sisyphus, beam_waist_sisyphus)
intensity = SisyphusLaser.get_intensity()
saturation_param = intensity/saturation_intensity
print(saturation_param)
