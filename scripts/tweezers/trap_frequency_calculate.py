# author: Marijn Venderbosch
# 2022-2024

import sys
import os
from scipy.constants import Boltzmann, proton_mass, epsilon_0, c, hbar, pi
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
h=hbar*2*pi
polarizability_ground_au=286 # atomic units
polarizability_3p1_au=195 # atomic units
mass=88*proton_mass
kHz=1e3
MHz=1e6

# variables
lamb=813e-9 # m
waist_diff_limited=0.8e-6 # m
diff_stark_measured_J=-3.4*MHz*h # J
number_traps = 5*5
beam_power = 340*1e-3*0.95/number_traps # W

# polarizatility in SI units
pol_atomic_units = Conversion().get_atomic_pol_unit()
polarizability_ground_si = polarizability_ground_au*pol_atomic_units

# The measured trap depth for the ground state in J
U0_measured_g_J = diff_stark_measured_J/(polarizability_3p1_au/polarizability_ground_au - 1)

# print trap depth in mK. 
print("measured U0 ground state: ", round(U0_measured_g_J/Boltzmann*1e3, 1), "mK or ",
    round(U0_measured_g_J/h/1e6, 1), "MHz")

# calculate theory stark shift
DiffractionLimitedTweezer = GaussianBeam(beam_power, waist_diff_limited)
intensity_diff_limited = DiffractionLimitedTweezer.get_intensity()

AtomCalc = AtomicCalculations(pol_atomic_units)
U0_theory_g_J = AtomCalc.ac_stark_shift(polarizability_ground_au, intensity_diff_limited)
print("diffraction limited U0 ground state: ", round(U0_theory_g_J/Boltzmann*1e3, 1),
    "mK or ", round(U0_theory_g_J/h/1e6, 1), "MHz")

# calculate waist from diff. limited vs measured trap depth
waist_measured = waist_diff_limited*np.sqrt(U0_theory_g_J/U0_measured_g_J)
print("estimated waist: ", round(waist_measured*1e6, 2), "um")

# calculate trap frequency 
trap_freq_rad = TrapFrequencies().trap_freq_radial(U0_measured_g_J, mass, waist_measured)
print("trap freq. radial: ", round(trap_freq_rad/2/np.pi/1e3), " kHz")

Tweezer = GaussianBeam(beam_power, waist_measured)
rayleigh_range_estimated = Tweezer.get_rayleigh_range(lamb)
trap_freq_axial = TrapFrequencies().trap_freq_axial(U0_measured_g_J, mass, rayleigh_range_estimated)
print("trap freq. axial: ", round(trap_freq_axial/2/np.pi/1e3), " kHz")
