# -*- coding: utf-8 -*-
"""
Created on Oct 8, 2022
@author: Marijn Venderbosch

based on the following example of the recoil-limited Sr88 MOT
https://python-laser-cooling-physics.readthedocs.io/en/latest/examples/MOTs/10_recoil_limited_MOT.html
"""

import numpy as np
import matplotlib.pyplot as plt
import pylcp
import scipy.constants as cts
from scipy.constants import pi, hbar
from pylcp.common import progressBar
import pathos
from matplotlib.patches import Ellipse


class RecoilLimitedMOT:
    """
    recoil limited Sr88 red MOT
    defines magnetic field, laser beams, governing equations and Hamiltonian
    """

    def __init__(self, detuning, saturation_parameter, length, time, alpha):
        self.detuning = detuning
        self.saturation = saturation_parameter
        self.length = length
        self.time = time
        self.alpha = alpha

    def magnetic_field(self):
        return pylcp.quadrupoleMagneticField(alpha)

    def laser_beams(self):
        return pylcp.conventional3DMOTBeams(delta=delf.detuning,
                                            s=self.saturation,
                                            beam_type=pylcp.infinitePlaneWaveBeam)

    def hamiltonian(self):
        return 


def main():
    #  independent variables
    wavenumber = 2 * np.pi * 689e-7  # 1/cm
    linewidth = 7.4e3  # 1/s, thus kHz linewidth
    detuning = -200/7.5
    saturation = 25
    bohr_magneton = cts.value('Bohr magneton in Hz/T')
    atomic_mass_unit = cts.value('atomic mass constant')

    #  dependent variables
    length = 1/wavenumber  # cm
    time = 1/linewidth  # s
    alpha = 1.5 * bohr_magneton * 1e-4 * 8 * length / (linewidth / (2 * pi))
    mass = 87.8 * atomic_mass_unit * (length * 1e-2)**2 / hbar / time
    gravity_vector = -np.array([0., 0., 9.8 * time**2 / (length * 1e-2)])

    print(mass, time, length, alpha, gravity_vector)

    #  create object
    sr_red_mot = RecoilLimitedMOT(detuning, saturation, length, time, alpha)


if __name__ == '__main__':
    main()
