# author: Marijn Venderbosch
# october 2022

"""script plots dipole radiation pattern for J_g=0 --> J_e=1 transition in Sr88 red and blue MOT"""

# %% imports

import matplotlib.pyplot as plt
import numpy as np

# %% variables

pi = np.pi
theta = phi = np.linspace(0, 2*pi, 1000)

# %% functions


class DipoleRadiationPattern:
    """functions that return dipole ratiation pattern for J_g = 0, J_e = 1
    from Madjarov2021 PhD thesis eq. (2.94,2.95)"""
    
    def __init__(self, angle):
        self.angle = theta
    
    def radiation_pattern_linear(self):
        return 3 / 8 / pi * np.sin(self.angle)**2
    
    def radiation_pattern_sigma(self):
        return 3 / 16 / pi * (1 + np.cos(self.angle)**2)


def main():
    dipole_radiation = DipoleRadiationPattern(theta)
    
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(projection='polar')

    ax.plot(theta, dipole_radiation.radiation_pattern_linear(),
            label=r"$m_e=0$")
    ax.plot(theta, dipole_radiation.radiation_pattern_sigma(),
            label=r"$m_e=\pm1$")

    fig.legend()
    plt.show()


if __name__ == "__main__":
    main()
