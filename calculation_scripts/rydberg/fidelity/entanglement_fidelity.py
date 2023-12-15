# author: Marijn Venderbosch
# 2023

"""
script computes 2 qubit gate fidelity errors from several dynamic decoherence
mechanisms. In order to do this, it computes rabi frequencies, lifetimes and 
interaction strengths as a function of n
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import proton_mass
import pandas as pd

from modules.atom_class import AtomicMotion, VanDerWaals, Polarizability
from modules.optics_class import Optics
from modules.rates_class import LightAtomInteraction
from modules.conversion_class import Conversion
from modules.noise_class import PhaseNoise, IntensityNoise, GateErrors, Lifetime

plt.style.use('default')

# %% variables

# fixed
pi = np.pi
mass_sr = 88*proton_mass  # [kg]
uv_wavelength = 316.6e-9  # [m]

# variable
temperature = 3e-6  # [K]
uv_beam_waist_y = 20e-6  # [m]
uv_beam_waist_x = 100e-6
trapdepth_K = 450e-6  # trap depth in [K] of optical tweezer
tweezer_waist = 0.82e-6  # [m], 813 nm 1/e^2 radius
beam_power = 100e-3  # power in rydberg laser at atoms in [W]
atom_separation_distance = 5e-6  # [m]
stddev_intensity = 0.008
stray_electric_fields = [0.001, 0.01, 0.1]  # [V/cm]

# n to consider
n_start = 50  # mainly limited by blockade radius
n_end = 80  # mainly limited by stray electric fields, polarizability
n_array = np.linspace(n_start, n_end, n_end-n_start+1)  
# %% get data


# ========================
# atomic motion
# ========================

wavenumber = 2*pi/uv_wavelength  # [1/m]
trap_frequency = AtomicMotion.trap_frequency_tweezer_radial(mass_sr, tweezer_waist, trapdepth_K)
variance_momentum, _ = AtomicMotion.doppler_broadening_tweezer(mass_sr, temperature, trap_frequency, uv_wavelength)
rydberg_intensity = Optics.cylindrical_gaussian_beam(uv_beam_waist_x, uv_beam_waist_y, beam_power)

# compute radial dipole matrix elements and rabi frequencies
rdmes = LightAtomInteraction.sr88_rdme_value_au(n_array)
rabi_freqs = Conversion.rdme_to_rabi(rdmes, rydberg_intensity, j_e=1)

# errors as a result of atomic motion leading to doppler broadening
# rabi frequency factor sqrt(2) enhanced for 2 atoms
motion_errors = GateErrors.atomic_motion_error(wavenumber, variance_momentum, mass_sr, np.sqrt(2)*rabi_freqs)


# ========================
# imperfect blockade
# ========================

# get van der waals interactions strenghts
c6_coefficients = []

for n in n_array:
    # n has to be int for ARC
    n = int(n)
    
    # get C6 coefficients and store result (n, L, J, m_j)
    c6 = VanDerWaals.calculate_c6_coefficients(n, 0, 1, 0)
    c6_coefficients.append(c6)

# get c6 coefficients in units of GHz/micron^6
c6_coefficients = np.array(c6_coefficients)*2*pi

# get interaction strength in Hz, convert atomic seperation to micron
interaction_strenghts = abs(c6_coefficients*1e9/(atom_separation_distance*1e6))

# errors as a result from imperfect blockade, convert interaction strength to Hz
blockade_errors = GateErrors.imperfect_blockade_error(np.sqrt(2)*rabi_freqs, interaction_strenghts)


# ========================
# phase noise, commented for now, because we don't know the phase noise spectrum yet
# ========================

# phasenoise_data = np.array(pd.read_csv('data/frequencynoise.csv', delimiter=';', decimal=',', header=None))

# # convert fourier freq to angular freq.
# omegas = phasenoise_data[:, 0]*2*pi
# phasenoise = phasenoise_data[:, 1]

# # error as a result of phase noise
# phasenoise_errors = []

# for rabifreq in rabi_freqs:
#     phase_error = PhaseNoise.compute_fidelity_error(rabifreq, omegas, phasenoise)
#     phasenoise_errors.append(phase_error)

# phasenoise_errors = np.array(phasenoise_errors)


# ========================
# intensity noise
# ========================

# commented, only assuming shot to shot noise
# intensitynoise_data = np.array(pd.read_csv('data/intensitynoise.csv', delimiter=';', decimal=',', header=None))
# # convert fourier freq to angular freq.
# omegas_intensity = intensitynoise_data[:, 0]*2*pi
# intensitynoise = intensitynoise_data[:, 1]
# plt.plot(omegas_intensity, intensitynoise)

intensity_errors = []

for rabifreq in rabi_freqs:
    intensity_error = IntensityNoise.shot_to_shot_fidelity_error(stddev_intensity)
    intensity_errors.append(intensity_error)

intensity_errors = np.array(intensity_errors)


# ========================
# lifetime
# ========================

# integrate rabi pupulation from t=0 to t=pi/(sqrt2*Omega)
time_spent_rydberg = pi/2*1/rabi_freqs

lifetimes = Lifetime.rydberg_state(n_array)
loss_errors = 1-np.exp(-time_spent_rydberg/lifetimes)

# =======================
# electric fields, commented for now, because we don't know what the stra electric field strength is
# =======================

polarizabilities = Polarizability.sr88_3s1(n_array)

# compute dc stark shift in [Hz] instead of [MHz] for several values
#%%
# empty list to loop over

dc_stark_errors_list = []
for electric_field in stray_electric_fields:
    dc_stark_shifts = Conversion.dc_stark_shift(polarizabilities, electric_field)*1e6
    dc_stark_errors = dc_stark_shifts**2/(np.sqrt(2)*rabi_freqs)**2
    dc_stark_errors_list.append(dc_stark_errors)


# %% plotting

# total error
total_errors = motion_errors + blockade_errors + loss_errors + intensity_errors  #+ dc_stark_errors + phase_noise_errors

fig, ax = plt.subplots(figsize = (4.5,3.5))
ax.plot(n_array, np.sqrt(2)*rabi_freqs/(2*pi*1e6))
ax.set_xlabel('n')
ax.set_ylabel('Blockaded Rabi freq. $\Omega_2/2\pi$ [MHz]')
plt.savefig('enhanced_rabi_freqs.png', bbox_inches='tight', dpi=400)

fig2, ax2 = plt.subplots(figsize=(4,3))
ax2.set_title(f'$R$={atom_separation_distance*1e6} $\mu$m')
ax2.plot(n_array, interaction_strenghts/(2*pi*1e9))
ax2.set_xlabel('n')
ax2.set_ylabel('$V_{DD}/2\pi$ [GHz]')

fig3, ax3 = plt.subplots()
ax3.plot(n_array, lifetimes*1e6)
ax3.set_xlabel('$n$')
ax3.set_ylabel('Rydberg state lifetime [$\mu$s]')

fig4, ax4 = plt.subplots(figsize=(4.5, 3.5))
ax4.plot(n_array, motion_errors, '-x', label='atomic motion', alpha=0.5)
#ax4.plot(n_array, blockade_errors, '-o', label='imperfect blockade', alpha=0.5), commented becaues negligible for n>50
#ax4.plot(n_array, phasenoise_errors, '-v', label='phase noise', alpha=0.5)
ax4.plot(n_array, loss_errors, '-^', label='rydberg state decay', alpha=0.5)
#ax4.plot(n_array, dc_stark_errors, '-<', label='stray electric fields', alpha=0.5)
ax4.plot(n_array, intensity_errors, '->', label='intensity noise', alpha=0.5)
ax4.plot(n_array, total_errors, '-s', label='total')
ax4.set_xlabel('$n$')
ax4.set_ylabel(r'Etanglement error, $1-\mathcal{F}$')
ax4.set_yscale('log')
ax4.legend(title='error, $1-\mathcal{F}$')
plt.savefig('entangling_fidelities.png', bbox_inches='tight', dpi=400)

fig5, ax5 = plt.subplots(figsize=(4.5, 3.5))
ax5.plot(n_array, dc_stark_errors_list[0], '-x', label=f'{stray_electric_fields[0]}')
ax5.plot(n_array, dc_stark_errors_list[1], '->', label=f'{stray_electric_fields[1]}')
ax5.plot(n_array, dc_stark_errors_list[2], '-<',label=f'{stray_electric_fields[2]}')
ax5.set_xlabel('$n$')
ax5.set_ylabel(r'Etanglement error, $1-\mathcal{F}$')
ax5.set_yscale('log')
ax5.legend(title='Stray electric field [V/cm]')
plt.savefig('stray_field_fidelities.png', bbox_inches='tight', dpi=400)
