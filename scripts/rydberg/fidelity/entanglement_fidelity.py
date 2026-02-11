# author: Marijn Venderbosch
# 2023

"""
script computes 2 qubit gate fidelity errors from several dynamic decoherence
mechanisms. In order to do this, it computes rabi frequencies, lifetimes and 
interaction strengths as a function of n
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import proton_mass, pi, hbar
import sys
import os

# Import custom modules
from modules.atom_class import Rydberg, Sr
from modules.atoms_tweezer_class import AtomicMotion
from modules.optics_class import GaussianBeam
from modules.conversion_class import Conversion
from modules.laser_class import AtomLightInteraction
from utils.units import uK, um, mW, h, nm

plt.style.use('default')
os.system('cls' if os.name == 'nt' else 'clear')

# variables
mass_sr = 88*proton_mass  # [kg]
uv_wavelength = 316.6*nm  # [m]
atom_temperature = 3*uK  # [K]
uv_beam_waist = 18*um  # [m], see madjarov 2020
trapdepth_K = 50*uK  # trap depth in [K] of optical tweezer, see madjarov 2020
tweezer_waist = 0.82*um  # [m], 813 nm 1/e^2 radius
uv_beam_power = 30*mW  # power in rydberg laser at atoms in [W], see madjarov 2020
atom_separation_distance = 4*um  # [m]
stddev_intensity = 0.008
stray_electric_fields = [0.001, 0.01, 0.1]  # [V/cm]

# principal quantum nr to consider
n_start, n_end = 50, 80
n_array = np.linspace(n_start, n_end, n_end - n_start + 1)  

# compute rabi frequencies
RydbergBeam = GaussianBeam(uv_beam_power, uv_beam_waist)
rydberg_intensity = RydbergBeam.get_intensity()
rabi_freqs = AtomLightInteraction.calc_rydberg_rabi_freq(n_array, rydberg_intensity, j_e=1)


def calc_intensity_infidelity_shotshot(sigma):
    """calculate infidelity as a result of intensity noise for shot to shot only

    Args:
        sigma (float): std dev in shot to shot intensity

    Returns:
        infidelity: 
    """
    fidelity = 0.5*(1 + np.exp(-0.5*pi**2*sigma**2))
    infidelity = 1 - fidelity
    return infidelity


# ========================
# noise sources
# ========================

## compute doppler broadening contribution
wavenumber = Conversion.wavenr_from_wavelength(uv_wavelength)
trap_frequency = AtomicMotion.trap_frequency_radial(mass_sr, tweezer_waist, trapdepth_K)
variance_momentum, _ = AtomicMotion.doppler_broadening_tweezer(mass_sr, atom_temperature, trap_frequency, uv_wavelength)
motion_errors = 5*wavenumber**2*variance_momentum/(4*mass_sr**2*rabi_freqs**2)

## imperfect blockade
# get interaction strength in Hz
interaction_strenghts_Hz = np.array(
    [Rydberg().calculate_interaction_strength(atom_separation_distance, int(n)) for n in n_array]
)
blockade_errors = (hbar*rabi_freqs)**2/(2*abs(h*interaction_strenghts_Hz)**2)

## intensity noise
intensity_errors = np.array(
    [calc_intensity_infidelity_shotshot(stddev_intensity) for rabi in rabi_freqs]
)

## finite rydberg state lifetime
# integrate rabi pupulation from t=0 to t=pi/(sqrt2*Omega)
time_spent_rydberg = pi/2*1/rabi_freqs
lifetimes = Sr.calc_rydberg_lifetime(n_array)
loss_errors = 1 - np.exp(-time_spent_rydberg/lifetimes)

## stray electric fields
polarizabilities = Sr.calc_polarizability_3s1(n_array)

dc_stark_2d_array = np.zeros(len(rabi_freqs), len(stray_electric_fields))

dc_stark_errors_list = np.array([
    AtomLightInteraction.calc_dc_stark_shift(polarizabilities, e_field)**2/(np.sqrt(2)*rabi_freqs)**2
    for e_field in stray_electric_fields
])

## plot all noise contributions invididually and the total error
# pick one electric field value
total_errors = motion_errors + blockade_errors + loss_errors + intensity_errors + dc_stark_errors

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(n_array, motion_errors, label='atomic motion', alpha=0.5)
ax.scatter(n_array, blockade_errors, label='imperfect blockade', alpha=0.5)
ax.scatter(n_array, loss_errors, label='rydberg state decay', alpha=0.5)
ax.scatter(n_array, dc_stark_errors, label='stray electric fields', alpha=0.5)
ax.scatter(n_array, intensity_errors, label='intensity noise', alpha=0.5)
ax.scatter(n_array, total_errors, label='total')
ax.set_xlabel(r'$n$')
ax.set_ylabel(r'Entanglement error, $1-\mathcal{F}$')
ax.set_yscale('log')
ax.legend()
plt.savefig('output/entangling_fidelities.png', bbox_inches='tight', dpi=400)

# plot DC stark effect on infidelity
fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
for idx in range(3):
    ax2.scatter(n_array, dc_stark_errors_list[idx], label=f'{stray_electric_fields[idx]}')
ax2.set_xlabel(r'$n$')
ax2.set_ylabel(r'Etanglement error, $1-\mathcal{F}$')
ax2.set_yscale('log')
ax2.legend(title='Stray electric field [V/cm]')
plt.savefig('output/stray_field_fidelities.png', bbox_inches='tight', dpi=400)

plt.show()
