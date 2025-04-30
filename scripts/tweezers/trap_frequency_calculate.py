# author: Marijn Venderbosch
# 2022-2024

import sys
import os
from scipy.constants import Boltzmann, proton_mass, hbar, pi
import numpy as np

# append path with 'modules' dir in parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries 
from atoms_tweezer_class import TrapFrequencies 
from conversion_class import Conversion
from optics_class import GaussianBeam
from atoms_tweezer_class import AtomicCalculations

# constants
h = hbar*2*pi
polarizability_ground_au = 286 # atomic units
polarizability_3p1_au = 195 # atomic units
mass = 88*proton_mass
kHz = 1e3
MHz = 1e6

# variables
lamb = 813e-9 # m
waist_diff_limited = 0.8e-6 # m
diff_stark_measured_J = -3.37*MHz*h # J
number_traps = 5*5
tweezer_power_total = 340*1e-3 # W
tweezer_power_total_reduced = 70*1e-3 # W for parametric heating measurements we ramp down traps
tweezer_power = tweezer_power_total*0.95/number_traps # W
print("power in tweezer: ", round(tweezer_power/1e-3, 1), "mW")

# polarizatility in SI units
pol_atomic_units = Conversion().get_atomic_pol_unit()
polarizability_ground_si = polarizability_ground_au*pol_atomic_units

# The measured trap depth for the ground state in J
U0_measured_g_J = diff_stark_measured_J/(polarizability_3p1_au/polarizability_ground_au - 1)
U0_measured_g_mK = U0_measured_g_J/Boltzmann/1e-3 # convert to mK
U0_measured_g_MHz = U0_measured_g_J/h/MHz # convert to MHz

# print trap depth in mK. 
print("measured U0 |g>: ", round(U0_measured_g_mK, 2), "mK or ", round(U0_measured_g_MHz, 1), "MHz")

# calculate theory stark shift
DiffractionLimitedTweezer = GaussianBeam(tweezer_power, waist_diff_limited)
intensity_diff_limited = DiffractionLimitedTweezer.get_intensity()
AtomCalc = AtomicCalculations(pol_atomic_units)
U0_theory_g_J = AtomCalc.ac_stark_shift(polarizability_ground_au, intensity_diff_limited)

# calculate waist from diff. limited vs measured trap depth
waist_measured = waist_diff_limited*np.sqrt(U0_theory_g_J/U0_measured_g_J)
print("estimated waist: ", round(waist_measured*MHz, 2), "um")

# calculate trap frequency for full power
trap_freq_rad = TrapFrequencies().trap_freq_radial(U0_measured_g_J, mass, waist_measured)
print("trap freq. radial full power: ", round(trap_freq_rad/kHz/(2*pi)), " kHz")

# calculate trap freq. for reduced tweezer power, compare againts parametric heating
trap_freq_rad_reduced = trap_freq_rad*np.sqrt(tweezer_power_total_reduced/tweezer_power_total)
print("trap freq. radial reduced power: ", round(trap_freq_rad_reduced/kHz/(2*pi)), " kHz")

# calculate axial trap freq. for full power
Tweezer = GaussianBeam(tweezer_power, waist_measured)
rayleigh_range_estimated = Tweezer.get_rayleigh_range(lamb)
trap_freq_axial = TrapFrequencies().trap_freq_axial(U0_measured_g_J, mass, rayleigh_range_estimated)
print("trap freq. axial: ", round(trap_freq_axial/kHz/(2*pi)), " kHz")
