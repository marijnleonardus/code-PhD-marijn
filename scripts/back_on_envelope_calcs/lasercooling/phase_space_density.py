import numpy as np
from scipy.constants import hbar, pi, Boltzmann, proton_mass

atomic_density_cm3 = 2e9  # atoms/cm^3
temperature = 5e-6  # K

mass = 88*proton_mass
atomic_density = atomic_density_cm3*1e6  # atoms/m^3


def phase_space_density(atomic_density, temperature):
    thermal_debroglie = hbar*np.sqrt(2*pi/(mass*Boltzmann*temperature))
    rho = atomic_density*thermal_debroglie**3
    return rho


print(phase_space_density(atomic_density, temperature))
