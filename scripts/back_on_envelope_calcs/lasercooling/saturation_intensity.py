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
beam_power_sisyphus = 0.03*1e-3  # W

saturation_intensity = Conversion.saturation_intensity(lifetime, wavelength)  # W/m^2

# compute saturation intensity Sisyphus laser
fiber_na = 0.08
f_collimationlens = 30e-3  # m
beam_waist_sisyphus = fiber_na*f_collimationlens  # m
print('beam waist sisyphus: ' + str(beam_waist_sisyphus))

SisyphusLaser = GaussianBeam(beam_power_sisyphus, beam_waist_sisyphus)
saturation_param_sishyphus = SisyphusLaser.get_intensity()/saturation_intensity
print('saturation param sisyphus: ' + str(saturation_param_sishyphus)) 

# compute power broadened linewidth
power_broadened_linewidth = 7.4e3*np.sqrt(1 + saturation_param_sishyphus)
print('linewidth: ' + str(power_broadened_linewidth))


# compute sat. param. red MOT
beam_power_red_mot = 3e-6  # W
beam_waist_red_mot = 3.2e-3  # m
RedMOTLaser = GaussianBeam(beam_power_red_mot, beam_waist_red_mot)
redmot_intensity = RedMOTLaser.get_intensity()
print(redmot_intensity)
saturation_param_red_mot = redmot_intensity/saturation_intensity
print(saturation_param_red_mot)