# author: Marijn L. Venderbosch
# September 18, 2022

# %% imports
from scipy.constants import Boltzmann, c, pi, hbar, Planck, atomic_mass

# %% global variables

optical_power = 1e-3  # W
beam_waist = 0.8e-6  # m
laser_wavelength = 820e-9  # m

linewidth_d1 = 2 * pi * 5.7e6  # 1/s
laser_wavelength_d1 = 795e-9  # m

linewidth_d2 = 2 * pi * 6.0e6  # 1/s
laser_wavelength_d2 = 780e-9  # m

# %% functions, classes


class DipolePotential:

    # constructor
    def __init__(self, power, waist, wavelength, wavelength_d1, wavelength_d2):
        self.power = power
        self.waist = waist
        self.wavelength = wavelength
        self.wavelength_d1 = wavelength_d1
        self.wavelength_d2 = wavelength_d2

    def return_intensity(self):
        return self.power / (pi * self.waist**2)

    def return_detuning_d1(self):
        return 2 * pi * c * (1 / wavelength - 1 / wavelength_d1)

    def return_detuning_d2(self):
        return 2 * pi * c * (1 / wavelength - 1 / wavelength_d2)

    def saturation_intensity(self):
        return 2 * pi**2 * hbar * c * linewidth


# %% objects
result = DipolePotential(optical_power, beam_waist)
print(result.return_intensity())
