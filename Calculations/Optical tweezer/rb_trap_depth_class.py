# author: Marijn L. Venderbosch
# September 18, 2022

# %% imports
import numpy as np

# %% variables

optical_power = 1e-3
beam_waist = 0.8e-6
pi = np.pi

# %% functions


class DipolePotential:

    # constructor
    def __init__(self, power, waist):
        self.power = power
        self.waist = waist

    def return_intensity(self):
        return self.power / (pi * self.waist**2)


result = DipolePotential(optical_power, beam_waist).return_intensity()
print(result)
