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

