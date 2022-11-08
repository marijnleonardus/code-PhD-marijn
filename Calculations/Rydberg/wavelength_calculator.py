# author: Marijn Venderbosch
# november 2022

"""script computes """

# %% imports

from scipy.constants import electron_mass, hbar, c, pi, e, proton_mass
from arc import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#  import Sr data from ARC database
Sr88 = Strontium88()

#  customize plot settings
matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

# %% variables

n_lower = int(40)
n_higher = int(70)
n_span = n_higher - n_lower 
n_array = np.linspace(n_lower, n_higher, n_span + 1)

sr_mass = 88  # amu

rydberg_constant = (1 - electron_mass / 88 / sr_mass) * C_Rydberg

# %% calculations

# Rydberg

rydberg_energy_list = []


def get_rydberg_energy(input_array):
    pre_factor = - 2 * pi * hbar * c * rydberg_constant
    
    for n in input_array:
        defect_n = Sr88.getQuantumDefect(n, l=0, j=1)
        energy_n = pre_factor / (n - defect_n)**2
        rydberg_energy_list.append(energy_n)
        
        rydberg_energy_array = np.array(rydberg_energy_list)
    return rydberg_energy_array


rydberg_energy_joule = get_rydberg_energy(n_array)

# Clock

clock_energy_joule = Sr88.getEnergy(5, 1, 0, s=1) * e  # J

# Transition

transition_energy_joule = rydberg_energy_joule - clock_energy_joule  # J
rydberg_energy_eV = rydberg_energy_joule / e  # eV

wavelength_array = 2 * pi * hbar * c / transition_energy_joule  # J
wavelength_difference = np.diff(wavelength_array) * 1e9  # nm

print("min. wavelength hop:" + str(np.min(wavelength_difference)))
print("max. wavelegnth hop:" + str(np.max(wavelength_difference)))

# %% plotting
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               figsize=(7, 3))
plt.subplots_adjust(wspace=0.5)

ax1.grid()
ax1.plot(n_array, rydberg_energy_eV)
ax1.set_xlabel('principal quantum number $n$')
ax1.set_ylabel('energy [J]')

ax2.grid()
ax2.plot(n_array, wavelength_array * 1e9)
ax2.set_xlabel('principal quantum number $n$')
ax2.set_ylabel(r'$\lambda$ [nm]')
ax2.axvspan(51, 65, color = 'red', alpha=0.2)

plt.savefig('rydberg_energies.pdf',
            bbox_inches='tight',
            dpi=300)
plt.show()
