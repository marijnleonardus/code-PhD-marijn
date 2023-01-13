# author: Marijn Venderbosch
# november 2022

"""script for computing the trap depths we can reach for Sr88 atoms
computes trap depth using polarizability data from Madjarov thesis as confirmed
by Robert de Keijzers's script

Takes into account losses along the way to the atoms"""

# %% imports, variables

from scipy.constants import pi, epsilon_0, c, Boltzmann
import units_script
import numpy as np

# %% variables

# Sr 88
polarizability = 286  # [au], atomic units at 813 nm, source: Madjarov thesis

# tweezer parameters
beam_waist = 0.82e-6  # [m]
number_tweezers = 7**2
laser_power = 0.9  # [W], laser power out of the fiber
aod_trap_depth_relative = 10  # how deep AOD movable is compared to SLM trap

# power losses along the way
diffraction_efficiency = 0.75
double_pass_efficiency = 0.7
fiber_coupling_efficiency = 0.5


# %% functions


def get_intensity(power, waist):
    """Intensity of Gaussian beam"""
     
    intensity = 2 * power / pi / waist**2
    return intensity
 
 
def get_power_at_atoms(array_size, power, aod_slm_ratio, double_pass_eff, coupl_eff, diffr_eff):
    """given laser power, compute power arriving at atoms by taking into account
    losses along the way, e.g. split into cross AOD path, SLM, microscope objective"""

    # compute fraction going to SLM beam path, which is split from crossed AOD path
    fraction_to_slm_path = array_size / (aod_slm_ratio + array_size)
    power_to_slm_path = power * fraction_to_slm_path

    # compute power after AOM and fiber coupling
    power_after_aom = power_to_slm_path * double_pass_eff * coupl_eff

    # compute power left after reflecting onto SLM, two losses:
    # 1) diffraction efficiency
    # 2) limited aperture size: some power lost that does not land on rectangular SLM aperture ~95%
    # for source latter see Venderbosch master thesis
    power_after_slm = power_after_aom * diffr_eff * 0.95

    # power after objective:
    # transmission of glass cell (uncoated) ~ 0.96
    # 10 optical elements with transmission ~ 0.99
    # microscope objective with transmission 0.42; source: Venderbosch master thesis
    power_to_objective = power_after_slm * 0.99**10 * 0.96
    power_after_objective = power_to_objective * 0.42
    return power_after_objective
 
 
def get_ac_stark(alpha):
    """compute AC Stark shift given intensity and polarizability alpha"""
     
    au = units_script.get_atomic_unit()
    power_atoms = get_power_at_atoms(number_tweezers,
                                     laser_power,
                                     aod_trap_depth_relative,
                                     double_pass_efficiency,
                                     fiber_coupling_efficiency,
                                     diffraction_efficiency)
    intensity = get_intensity(power_atoms, beam_waist)
     
    shark_shift = alpha * au / (2 * c * epsilon_0) * intensity
    return shark_shift
 
 
def get_trap_depth_mk(array_size):
    """compute trap depth per SLM site, in terms of equivalent temperature"""
     
    # trap depth in milliKelvin
    trap_depth = get_ac_stark(polarizability)
    trap_depth_mk = trap_depth / Boltzmann / 1e-3
     
    # trap depth per SLM site
    trap_temp_mk_per_trap = trap_depth_mk / array_size
    return trap_temp_mk_per_trap


def main():
    print("Depth per trap is: " + str(np.round(get_trap_depth_mk(number_tweezers), decimals=3)) + " mK")


if __name__ == "__main__":
    main()
    