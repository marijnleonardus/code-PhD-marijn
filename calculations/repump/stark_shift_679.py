# author: Marijn Venderbosch
# 2023

from arc import Strontium88
from functions.conversion_functions import (compute_rabi_freq,
                                            intensity_to_electric_field,
                                            compute_stark_shift)
from classes.optics_class import Optics
from classes.rates import LightAtomInteraction
Sr88 = Strontium88()

# %% variables

# repump detuning beam
wavelength_repump = 689e-9  # m
numerical_aperture = 0.5
power_repump = 1e-3  # W
detuning_repump = 10e9  # Hz
linewidth_repump = 9e6  # Hz

# %% execution



def get_rdme():
    
    # get ARC radial dipole matrix element from 3P0, mj=0 to (6sns) 3S1, mj = 0 
    # for linear polarization: q=0
    rdme = Sr88.getDipoleMatrixElement(5, 1, 0,
                                       0,
                                       6, 0, 1, 
                                       0,
                                       q=0, 
                                       s=1)
    return rdme


def print_stark_scatter(wavelength, power, numerical_aperture,  detuning, waist, linewidth):
    
    # compute diffraction limited waist when shining through microscope objective
    waist = Optics.gaussian_beam_diffraction_limit(wavelength, numerical_aperture)
    
    # compute intensity of the 'repump tweezer'
    intensity = Optics.gaussian_beam_intensity(waist, power)
    
    # convert intensity to electric field strengh in [V/m]
    electric_field_strength = intensity_to_electric_field(intensity)
    
    # get RDME in atomic units from 3P0, mj=0 to (6sns) 3S1, mj = 0 
    # for linear polarization: q=0
    rdme = get_rdme()
    
    # compute rabi frequency of transition
    rabi_freq = compute_rabi_freq(rdme, electric_field_strength)
    
    # compute stark shift
    stark_shift = compute_stark_shift(rabi_freq, detuning)
    print("Stark shift: " + str(stark_shift * 1e-9) + " GHz")
    
    # compute off-resonant scattering rate 
    repump_scatter_rate = LightAtomInteraction.scattering_rate_power(linewidth,
                                                                     detuning,
                                                                     wavelength,
                                                                     waist,
                                                                     power)
    
    print("Off resonant scattering rate: " + str(repump_scatter_rate) + " Hz")

print_stark_scatter(wavelength_repump, power_repump, numerical_aperture, power_repump, detuning_repump, linewidth_repump)
