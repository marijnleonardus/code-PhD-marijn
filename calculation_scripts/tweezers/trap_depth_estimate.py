# author: Marijn Venderbosch
# november 2022

"""script for computing the trap depths we can reach for Sr88 atoms
computes trap depth using polarizability data from Madjarov thesis as confirmed
by Robert de Keijzers's script.
Takes into account losses along the way to the atoms"""

# stock libraries
from scipy.constants import epsilon_0, c, Boltzmann, hbar, pi
import numpy as np

# user defined libraries
from modules.optics import GaussianBeam
from modules.atoms import AtomicCalculations
from modules.units import UnitConversion

# %% variables

# Sr 88
polarizability = 286  # [au], atomic units at 813 nm, source: Madjarov thesis

# tweezer parameters
beam_waist = 0.82e-6  # [m]
array_dim = 10  # how big square tweezer array
nr_tweezers = int(array_dim**2)
laser_power = 0.9  # [W], laser power out of the fiber
movable_tw_depth = 10  # how deep AOD movable is compared to SLM trap

# %% functions

def get_power_at_atoms(array_size, power, aod_slm_ratio):
    """given laser power, compute power arriving at atoms by taking into account
    losses along the way, e.g. split into cross AOD path, SLM, microscope objective"""

    diffr_eff = 0.75
    double_pass_eff= 0.7
    coupl_eff = 0.5
    slm_aperture_loss = 0.95
    obj_trans = 0.42
    optics_refl = 0.99
    uncoated_refl = 0.96

    # compute fraction going to SLM beam path, which is split from crossed AOD path
    total_frac = aod_slm_ratio + array_size
    frac_slm = array_size/total_frac
    power_slm_path= power*frac_slm

    # compute power after AOM and fiber coupling
    power_after_aom = power_slm_path*double_pass_eff*coupl_eff

    # compute power left after reflecting onto SLM. Loss from diffraction efficiency
    # and limited aperture size: gaussian beam not fully reflecting of rectangular SLM
    power_after_slm = power_after_aom*diffr_eff*slm_aperture_loss

    # loss as a a result of glass cell (uncoated) and optical elements (coated)
    # microscope objective with transmission 0.42; source: Venderbosch master thesis
    power_to_obj = power_after_slm*optics_refl**10*uncoated_refl
    slm_power_atoms = power_to_obj*obj_trans

    # compute AOD beam power
    power_aod_path = power - power_slm_path
    
    # losses are the same for AOD beampath, so can directly compute same ratio
    movable_tw_power = power_aod_path*slm_power_atoms/power_slm_path
    
    return slm_power_atoms, movable_tw_power
 
def get_trap_depth_mk(ac_stark):
    """
    compute trap depth in mK

    Args:
        ac_stark (float): trap depth in J

    Returns:
       trap_depth_mk: trap depth in mK
    """
    # trap depth in K
    trap_depth_k = ac_stark / Boltzmann
    trap_depth_mk = trap_depth_k / 1e-3
    return trap_depth_mk

# %% main

 
def main():
    # atomic unit
    Units = UnitConversion()
    bohr_radius = Units.get_bohr_radius()
    hartree_unit = Units.get_hartree_unit(bohr_radius)
    au = Units.get_atomic_unit(bohr_radius, hartree_unit)

    # compute power left after loses
    power_atoms, movable_tw_power = get_power_at_atoms(nr_tweezers, laser_power, movable_tw_depth)
    power_per_atom = power_atoms/nr_tweezers

    # compute intensity
    SLMTweezer = GaussianBeam(power_per_atom, beam_waist)
    slm_intensity = SLMTweezer.get_intensity()

    # compute AC stark shift for SLM tweezer
    Atom = AtomicCalculations(au)
    ac_stark_slm = Atom.ac_stark_shift(polarizability, slm_intensity) 

    # compute trap depth in mK for SLM tweezer
    trap_depth_mk = get_trap_depth_mk(ac_stark_slm)
    print("Depth per trap is: " + str(np.round(trap_depth_mk, decimals=3)) + " mK")

    # compute light shift for AOD movable tweezer
    AODBeam = GaussianBeam(movable_tw_power, beam_waist)
    aod_intensity = AODBeam.get_intensity()
    planck = 2*pi*hbar
    lightshift_aod = Atom.ac_stark_shift(polarizability, aod_intensity)
    ac_stark_shift_hz = lightshift_aod/planck
    print("AC stark shift is: " + str(np.round(ac_stark_shift_hz * 1e-6,1)) + " MHz")


if __name__ == "__main__":
    main()
    