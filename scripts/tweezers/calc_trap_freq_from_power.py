# author: Marijn Venderbosch
# 2022-2025

import sys
import os
from scipy.constants import Boltzmann, proton_mass
import numpy as np

# append path with 'modules' dir in parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

utils_dir = modules_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))
sys.path.append(utils_dir)

# user defined libraries 
from atoms_tweezer_class import AtomicMotion 
from optics_class import GaussianBeam
from atoms_tweezer_class import AtomicCalculations
from units import (h, MHz, mK, mW, pol_1s0, pol_1s0_au, pol_3p1_mj0, atomic_pol_unit)

# constants
mass = 88*proton_mass

# variables
lamb = 813.4e-9 # m
waist_diff_limited = 0.8e-6 # m
diff_stark_measured_J = +3.37*MHz*h # J
number_traps = 5*5
tw_power_set = 340*mW
refl_coeff = 0.96
tw_power_total = tw_power_set*refl_coeff**2 # W
tw_power_single = tw_power_total/number_traps # W

# %% compute waist from power and trap depth

print('compute waist from power and trap depth')
print('======================================================')

# The measured trap depth for the ground state 1S0
U0_g = diff_stark_measured_J/(1 - pol_3p1_mj0/pol_1s0)
print("measured U0 |g>: ", round(U0_g/(Boltzmann*mK), 3), "mK or ", round(U0_g/(h*MHz), 2), "MHz")

# calculate theory stark shift
DiffractionLimitedTweezer = GaussianBeam(tw_power_single, waist_diff_limited)
intensity_diff_limited = DiffractionLimitedTweezer.get_intensity()
AtomCalc = AtomicCalculations(atomic_pol_unit)
U0_theory_g_J = AtomCalc.ac_stark_shift(pol_1s0_au, intensity_diff_limited)
print("U0 diff limited |g>: ", round(U0_theory_g_J/(Boltzmann*mK), 3), " mK or ", round(U0_theory_g_J/(h*MHz), 2), "MHz")

# calculate waist from diff. limited vs measured trap depth
waist_measured = waist_diff_limited*np.sqrt(U0_theory_g_J/U0_g)
print("estimated waist: ", round(waist_measured*MHz, 2), "um")
