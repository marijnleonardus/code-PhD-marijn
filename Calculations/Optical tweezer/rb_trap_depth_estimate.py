# -*- sat_I1=None-8 -*-
"""
Created on Thu Dec 16 09:54:40 2021
@author: Marijn Venderbosch
Computes trap depth from eq. 1.6 PhD thesis Ludovic Brossard
"""

# %% Imports

from scipy.constants import Boltzmann, c, pi, hbar, Planck, atomic_mass
import numpy as np

# %% Variables

mRb = 87 * atomic_mass

# Dipole trap
power = 55*.41/5*1e-3  # mW
waist = 0.8e-6  # m
rayleigh = 3.6e-6  # m
wavelength = 820e-9  # m

# D1 Rubidium
linewidth_d1 = 2 * pi * 5.7e6  # 1/s
line_wavelength_d1 = 795e-9  # m

# D2 Rubidium
linewidth_d2 = 2 * pi * 6e6  # 1/s
line_wavelength_d2 = 780e-9

# Functions


def intensity(optical_power, beam_waist):
    return 2 * optical_power / (pi * beam_waist**2)


def detuning(laser_wavelength, line_wavelength):
    return 2 * pi * c * (1 / laser_wavelength - 1 / line_wavelength)


def saturation_intensity(linewidth, line_wavelength):
    return 2 * pi**2 * hbar * c * linewidth / (3 * line_wavelength**3)


def dipole_potential(det1, det2,
                     linewidth1, linewidth2,
                     sat_intensity1, sat_intensity2):
    # eq. 1.6 from Brossard PhD thesis (Browaeys, 2020)
    # matrix elements same for all transitions: linear polarization
    # prefactors 1/3 and 2/3 from 2J'+1/2J+1 and Clebsch-Gordon

    d1_contribution = 1 * linewidth1**2 / (3 * sat_intensity1 * det1)
    d2_contribution = 2 * linewidth2**2 / (3 * sat_intensity2 * det2)

    pre_factor = hbar / 8 * (d1_contribution + d2_contribution)

    return pre_factor * intensity(waist, power)


def trap_frequency_radial(beam_waist, mass, potential):
    return np.sqrt(-4 * potential / (mass * beam_waist**2))


def trap_frequency_axial(mass, potential_depth):
    return np.sqrt(-2 * potential_depth / (mass * rayleigh**2))

# %% Executing functions and print result


detuning_d1 = detuning(wavelength, line_wavelength_d1)
detuning_d2 = detuning(wavelength, line_wavelength_d2)

saturation_intensity_d1 = saturation_intensity(linewidth_d1,
                                               line_wavelength_d1)

saturation_intensity_d2 = saturation_intensity(linewidth_d2,
                                               line_wavelength_d2)

dipole_potential = dipole_potential(detuning_d1, detuning_d2,
                                    linewidth_d1, linewidth_d2,
                                    saturation_intensity_d1,
                                    saturation_intensity_d2)

potential_depth_mK = round(-dipole_potential / Boltzmann * 1e3, 2)
print("Trap depth is: " + str(potential_depth_mK) + " mK")

potential_depth_MHz = round(-dipole_potential / Planck * 1e-6, 1)
print("Trap depth (Hz) is: " + str(potential_depth_MHz) + "MHz")

radial_trap_frequency = trap_frequency_radial(waist, mRb, dipole_potential)
radial_trap_frequency_kHz = round(radial_trap_frequency * 1e-3 / (2 * np.pi))

axial_trap_frequency = trap_frequency_axial(mRb, dipole_potential)
axial_trap_frequency_kHz = round(axial_trap_frequency * 1e-3 / (2 * np.pi))
print("Raxial, axial trap frequency are: " +
      str(radial_trap_frequency_kHz) + ", and " +
      str(axial_trap_frequency_kHz) + " (kHz * 2pi)")
