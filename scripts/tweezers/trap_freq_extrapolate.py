# author: Marijn Venderbosch
# 2022-2025

import sys
import os
from scipy.constants import proton_mass, pi
import numpy as np

# append path with 'modules' dir in parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

utils_dir = modules_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))
sys.path.append(utils_dir)

# user defined libraries 
from units import (h, MHz, kHz, um, mW)

# constants
mass = 88*proton_mass

# variables
lamb = 813.4e-9 # m
trap_depth_measured = 11.08*MHz*h
tw_power_set = 340*mW
refl_coeff = 0.96
tw_power_total = tw_power_set*refl_coeff**2

# radial trap freq. measurement
trap_freq_rad_measured = 51.3*kHz
power_tw_paramheating_rad_set = 120*mW

# axial trap freq. measurement
trap_freq_ax_measured = 5.74*kHz
power_tw_paramheating_ax_set = 70*mW

#%% compute waist from trap depth and trap freq measurement
print()
print('compute waist from trap depth and trap freq measurement')
print('======================================================')

# calculate trap freq. for full tweezer power
# for param heating measurement we ramp down the traps, ratio 'chi' in the paper
chi_rad = power_tw_paramheating_rad_set/tw_power_set
tw_pwr_tot_rad_reduced = chi_rad*tw_power_total 
trap_freq_rad_extrapolated = trap_freq_rad_measured*(chi_rad)**(-1/2)
print("trap freq. radial at full power, radial: ", round(trap_freq_rad_extrapolated/kHz, 3), " kHz")

# compute waist 
w0 = np.sqrt(4*trap_depth_measured/(mass*(2*pi*trap_freq_rad_extrapolated)**2))
print('calc. w0 : ', round(w0/um, 4), 'um')

# and now for axial direction
chi_ax = power_tw_paramheating_ax_set/tw_power_set
tw_pwr_tot_ax_reduced = chi_ax*tw_power_total 
trap_freq_ax_extrapolated = trap_freq_ax_measured*(chi_ax)**(-1/2)
print("trap freq. axial at full power, axial: ", round(trap_freq_ax_extrapolated/kHz, 3), " kHz")

# compute rayleigh range
zr = np.sqrt(2*trap_depth_measured/(mass*(2*pi*trap_freq_ax_extrapolated)**2))
print('calc. zr : ', round(zr/um, 4), 'um')
