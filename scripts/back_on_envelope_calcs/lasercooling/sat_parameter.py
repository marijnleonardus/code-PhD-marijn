import sys
import os
from scipy.constants import pi

# import module from '../modules'
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from optics_class import GaussianBeam
from laser_class import AtomLightInteraction
from units import MHz, mW, mm, nm

# variables
linewidth = 2*pi*32*MHz  # Hz
lifetime = 1/linewidth  # s
wavelength = 461*nm   # m
beam_power = 30e-3  # W

saturation_intensity = AtomLightInteraction().saturation_intensity(lifetime, wavelength)  # W/m^2
print('saturation intensity: ' + str(saturation_intensity) + ' W/m^2')

# compute saturation intensity Sisyphus laser
fiber_na = 0.08
f_collimationlens = 60*mm  # m
beam_waist = fiber_na*f_collimationlens  # m
print('beam waist: ' + str(beam_waist))

LaserBeam = GaussianBeam(beam_power, beam_waist)
saturation_param = LaserBeam.get_intensity()/saturation_intensity
print('saturation param: ' + str(saturation_param)) 
