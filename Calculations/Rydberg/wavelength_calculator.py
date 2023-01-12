# author: Marijn Venderbosch
# november 2022

from scipy.constants import electron_mass, hbar, c, pi, e
from arc import Strontium88, C_Rydberg
from classes.conversion_functions import rdme_to_rate, rabi_freq_to_rate, energy_to_wavelength
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#  import Sr data from ARC database
atom = Strontium88()

# %% variables

# range of principal quantum numbers n 
n_lower = int(40)
n_higher = int(100)
n_span = n_higher - n_lower 
n_array = np.linspace(n_lower, n_higher, n_span + 1)

# definition rydberg constant (see thesis Madjarov)
rydberg_constant = (1 - electron_mass / atom.mass) * C_Rydberg

wavelength_tuning_range = 0.07 * 1e-9 # m

# %% calculations

# Rydberg
def get_rydberg_energy(input_array):
    pre_factor = - 2 * pi * hbar * c * rydberg_constant
    
    # empty list
    rydberg_energy_list = []
    
    for n in input_array:
        defect_n = atom.getQuantumDefect(n, l=0, j=1)
        energy_n = pre_factor / (n - defect_n)**2
        rydberg_energy_list.append(energy_n)
        
        rydberg_energy_array = np.array(rydberg_energy_list)
    return rydberg_energy_array


rydberg_energy_joule = get_rydberg_energy(n_array)

# Clock
clock_energy_joule = atom.getEnergy(5, 1, 0, s=1) * e  # J

# Transition
transition_energy_joule = rydberg_energy_joule - clock_energy_joule  # J
rydberg_energy_eV = rydberg_energy_joule / e  # eV

wavelength_array = energy_to_wavelength(transition_energy_joule)  # J
wavelength_difference = np.diff(wavelength_array) * 1e9  # nm

# assume centered around n = 61
wavelength_n61 = wavelength_array[61 - n_lower]

# %% plotting

#  ARC messing with settings
matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

# ionizization energy
fig1, ax1 = plt.subplots()
ax1.grid()

ax1.plot(n_array, rydberg_energy_eV, label=r'$E-E_{ionize}$')

ax1.set_xlabel('$n$')
ax1.set_ylabel('energy [eV]')
ax1.legend()

# n vs lambda
fig2, ax2 = plt.subplots()
ax2.grid()

ax2.plot(n_array, wavelength_array * 1e9, label='Clock state to Rydberg')

ax2.set_xlabel('$n$')
ax2.set_ylabel(r'$\lambda$ [nm]')
ax2.legend()

# lambda vs n
fig3, ax3 = plt.subplots()
ax3.grid()

ax3.plot(wavelength_array * 1e9, n_array, label='Rydberg excitation wavelength')

ax3.set_xlabel(r'$\lambda$ [nm]')
ax3.set_ylabel('$n$')
ax3.set_ylim(40, 70)

# highlight particular section
ax3.axvspan((wavelength_n61 - wavelength_tuning_range / 2)*1e9, 
            (wavelength_n61 + wavelength_tuning_range / 2)*1e9,
            alpha=0.5, color='green',
            label='Fiber laser tuning range around $n=61$')
ax3.axhspan(58, 64,
            color='red', alpha=0.3, 
            label='accessible Rydberg states')
ax3.legend()

plt.show()
