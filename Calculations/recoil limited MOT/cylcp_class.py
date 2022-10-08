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
from pylcp.common import progressBar
import pathos
from matplotlib.patches import Ellipse


class RecoilLimitedMOT:
    """
    recoil limited Sr88 red MOT
    defines magnetic field, laser beams, governing equations and Hamiltonian
    """

    def __init__(self, detuning, saturation_parameter, linewidth, wavenumber, length, time):
        self.detuning = detuning
        self.saturation = saturation_parameter
        self.linewidth = linewidth
        self.wavenumber = wavenumber
        self.length = length
        self.time = time

    def magnetic_field(self):
        return pylcp.conventional3DMOTBeams(delta=delf.detuning,
                                            beam_type=pylcp.infinitePlaneWaveBeam)


def main():
    # variables
    detuning = -200/7.5
    saturation = 25
    wavenumber = 2 * np.pi * 689e-7  # 1/cm
    linewidth = 7.4e-3  # 1/s
    length = 1/wavenumber  # cm
    time = 1/linewidth  # s

    # create object
    sr_red_mot = RecoilLimitedMOT(detuning, saturation, linewidth, wavenumber, length, time)
    print(sr_red_mot)


if __name__ == '__main__':
    main()
