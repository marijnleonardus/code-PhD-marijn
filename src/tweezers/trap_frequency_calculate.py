import sys
import os
from scipy.constants import Boltzmann, proton_mass, epsilon_0, c
import numpy as np

# append path with 'modules' dir in parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries 
from atoms_tweezer import TrapFrequencies 
from units import UnitConversion

# %% variables
polarizability_au = 286 # au
power_trap = 5e-3 # W
trapdepth=35e-6*Boltzmann  # 35 uK
mass = 88*proton_mass
lamb = 813e-9

# %% compute polarizability

Units = UnitConversion()
bohr_radius = Units.get_bohr_radius()
hartree_unit = Units.get_hartree_unit(bohr_radius)
au = Units.get_atomic_unit(bohr_radius, hartree_unit)
polarizability_SI = polarizability_au*au

# %% calculate waist from trap depth and power

intensity = 2*epsilon_0*c*trapdepth/polarizability_SI
waist = np.sqrt(2*power_trap/(np.pi*intensity))
print(waist)

# %% calculate trap frequency 

TrapFreqs = TrapFrequencies()
trap_freq_rad = TrapFreqs.trap_freq_radial(trapdepth, mass, waist)
trap_freq_axial = TrapFreqs.trap_freq_axial(trapdepth, mass, np.pi*waist**2/lamb)

print(trap_freq_rad/2/np.pi)
print(trap_freq_axial/2/np.pi)
