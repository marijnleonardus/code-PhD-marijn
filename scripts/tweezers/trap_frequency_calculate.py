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
from atoms_tweezer_class import AtomicMotion 
from conversion_class import Conversion
from optics_class import GaussianBeam
from atoms_tweezer_class import AtomicCalculations
from units import atomic_pol_unit, h, kHz, MHz, mK 

# constants
pol_1s0_au = 286 # atomic units
pol_3p1_au = 199 # atomic units
mass = 88*proton_mass

# variables
lamb = 813.4e-9 # m
waist_diff_limited = 0.8e-6 # m
diff_stark_measured_J = -3.37*MHz*h # J
number_traps = 5*5
tweezer_power_total = 330*1e-3 # W
tweezer_power_total_reduced = 70*1e-3 # W for parametric heating measurements we ramp down traps
tweezer_power = tweezer_power_total*0.95/number_traps # W

# The measured trap depth for the ground state 1S0
pol_1s0_SI = pol_1s0_au*atomic_pol_unit
U0_measured_g_J = diff_stark_measured_J/(pol_3p1_au/pol_1s0_au - 1)
print("measured U0 |g>: ", round(U0_measured_g_J/(Boltzmann*mK), 3), "mK or ", round(U0_measured_g_J/(h*MHz), 2), "MHz")

# calculate theory stark shift
DiffractionLimitedTweezer = GaussianBeam(tweezer_power, waist_diff_limited)
intensity_diff_limited = DiffractionLimitedTweezer.get_intensity()
AtomCalc = AtomicCalculations(atomic_pol_unit)
U0_theory_g_J = AtomCalc.ac_stark_shift(pol_1s0_au, intensity_diff_limited)

# calculate waist from diff. limited vs measured trap depth
waist_measured = waist_diff_limited*np.sqrt(U0_theory_g_J/U0_measured_g_J)
print("estimated waist: ", round(waist_measured*MHz, 2), "um")

# calculate trap frequency for full power
trap_freq_rad = AtomicMotion().trap_frequency_radial(mass, waist_measured, U0_measured_g_J/Boltzmann)
print("trap freq. radial full power: ", round(trap_freq_rad/kHz/(2*pi)), " kHz")

# calculate trap freq. for reduced tweezer power, compare againts parametric heating
trap_freq_rad_reduced = trap_freq_rad*np.sqrt(tweezer_power_total_reduced/tweezer_power_total)
print("trap freq. radial reduced power: ", round(trap_freq_rad_reduced/kHz/(2*pi)), " kHz")

# calculate axial trap freq. for full power
Tweezer = GaussianBeam(tweezer_power, waist_measured)
rayleigh_range_estimated = Tweezer.get_rayleigh_range(lamb)
trap_freq_axial = AtomicMotion().trap_frequency_axial(mass, rayleigh_range_estimated, U0_measured_g_J/Boltzmann)
print("trap freq. axial: ", round(trap_freq_axial/kHz/(2*pi)), " kHz")
