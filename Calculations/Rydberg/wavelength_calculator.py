# author: Marijn Venderbosch
# november 2022

from scipy.constants import electron_mass, hbar, c, pi, e
from arc import Strontium88, C_Rydberg
from lib.conversion_class import Conversion
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#  import Sr data from ARC database
atom = Strontium88()

# %% variables

# range of principal quantum numbers n 
n_lower = int(40)
n_higher = int(100)
n_span = n_higher-n_lower 
n_array = np.linspace(n_lower, n_higher, n_span+1)

# definition rydberg constant (see thesis Madjarov)
rydberg_constant = (1-electron_mass/atom.mass)*C_Rydberg

tuning_range = 0.07*1e-9 # m

# %% calculations

# Rydberg
def rydberg_transition_wavelengths(n_list, reference_energy):
    
    # empty list
    wavelengths = []
    
    for n in n_list:
        defect = atom.getQuantumDefect(n, l=0, j=1, s=1)
        energy = -2*pi*hbar*c*rydberg_constant/(n-defect)**2
                
        transition_energies = energy-reference_energy  # [J]
        
        wavelength = Conversion.energy_to_wavelength(transition_energies)
        wavelengths.append(wavelength)
    return wavelengths


energy_3p0 = atom.getEnergy(5, 1, 0, s=1)*e  # J
energy_3p2 = atom.getEnergy(5, 1, 2, s=1)*e  # J

wavelengths_3p0 = np.array(rydberg_transition_wavelengths(n_array, energy_3p0))
wavelengths_3p2 = np.array(rydberg_transition_wavelengths(n_array, energy_3p2))


# %% plotting

#  ARC messing with settings
matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

# n vs lambda 3P0
fig1, ax1 = plt.subplots()
ax1.plot(n_array, wavelengths_3p0*1e9, label=r'${}^3P_0$ to Rydberg')
ax1.plot(n_array, wavelengths_3p2*1e9, label=r'${}^3P_2$ to Rydberg')
ax1.set_xlabel('$n$')
ax1.set_ylabel(r'$\lambda$ [nm]')
ax1.legend()

# lambda vs n for 3P0
fig2, ax2 = plt.subplots()
ax2.grid()
ax2.plot(wavelengths_3p0*1e9, n_array, label=r'${}^3P_0$ to Rydberg')
ax2.plot(wavelengths_3p2*1e9, n_array, label=r'${}^3P_2$ to Rydberg')
ax2.set_xlabel(r'$\lambda$ [nm]')
ax2.set_ylabel('$n$')
ax2.set_ylim(40, 70)
ax2.legend()

plt.show()
