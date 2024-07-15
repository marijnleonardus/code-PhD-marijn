# author: Marijn Venderbosch
# november 2022

"""script for computing the trap depths we can reach for Sr88 atoms
computes trap depth using polarizability data from Madjarov thesis as confirmed
by Robert de Keijzers's script.
Takes into account losses along the way to the atoms"""

# stock libraries
from scipy.constants import epsilon_0, c, Boltzmann, hbar, pi
import numpy as np
import sys
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'modules' directory
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))

# Add the 'modules' directory to the Python path
sys.path.append(modules_dir)

# user defined libraries
from optics import GaussianBeam
from atoms_tweezer import AtomicCalculations
from units import UnitConversion

# %% variables

# Sr 88
polarizability = 286  # [au], atomic units at 813 nm, source: Madjarov thesis

# tweezer parameters
beam_waist = 0.0.9e-6  # [m]
array_dim = 10  # how big square tweezer array
nr_tweezers = int(array_dim**2)
power_at_atoms = 0.16  # [W], laser power out of the fiber

# %% functions

 
def get_trap_depth_mk(ac_stark):
    """
    compute trap depth in mK

    Args:
        ac_stark (float): trap depth in J

    Returns:
       trap_depth_mk: trap depth in mK
    """
    trap_depth_kelvin = ac_stark/Boltzmann
    trap_depth_mk = trap_depth_kelvin/1e-3
    return trap_depth_mk

 
def main():
    # get atomic unit
    bohr_radius = UnitConversion().get_bohr_radius()
    hartree_unit = UnitConversion().get_hartree_unit(bohr_radius)
    au = UnitConversion().get_atomic_unit(bohr_radius, hartree_unit)

    # compute power left after loses
    power_per_atom = power_atoms/nr_tweezers

    # compute intensity
    SLMTweezer = GaussianBeam(power_per_atom, beam_waist)
    slm_intensity = SLMTweezer.get_intensity()

    # compute AC stark shift for SLM tweezer
    SrAtom = AtomicCalculations(au)
    ac_stark_slm = SrAtom.ac_stark_shift(polarizability, slm_intensity) 

    # compute trap depth in mK for SLM tweezer
    trap_depth_mk = get_trap_depth_mk(ac_stark_slm)
    print("Depth per trap is: " + str(np.round(trap_depth_mk, decimals=3)) + " mK")


if __name__ == "__main__":
    main()
    