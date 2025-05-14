import sys
import os
from scipy.constants import pi

# import module from '../modules'
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from optics_class import GaussianBeam
from laser_class import AtomLightInteraction

# variables
linewidth = 2*pi*7.4e3  # Hz
lifetime = 1/linewidth  # s
wavelength = 689e-9   # m
beam_power_sisyphus = 0.2*1e-3  # W

saturation_intensity = AtomLightInteraction.saturation_intensity(lifetime, wavelength)  # W/m^2

# compute saturation intensity Sisyphus laser
fiber_na = 0.08
f_collimationlens = 30e-3  # m
beam_waist_sisyphus = fiber_na*f_collimationlens  # m
print('beam waist sisyphus: ' + str(beam_waist_sisyphus))

SisyphusLaser = GaussianBeam(beam_power_sisyphus, beam_waist_sisyphus)
saturation_param_sishyphus = SisyphusLaser.get_intensity()/saturation_intensity
print('saturation param sisyphus: ' + str(saturation_param_sishyphus)) 
