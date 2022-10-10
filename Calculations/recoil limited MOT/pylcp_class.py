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


class RecoilLimitedMOT:
    """
    recoil limited Sr88 red MOT
    defines magnetic field, laser beams, gravity vector and hamiltonian
    """

    # initialize variables. length and time are characteristic length/time scales
    def __init__(self, detuning, saturation_parameter, length, time, alpha, mass):
        self.detuning = detuning
        self.saturation = saturation_parameter
        self.length = length
        self.time = time
        self.alpha = alpha
        self.mass = mass

    # quadrupole magnetic field defined by alpha
    def magnetic_field(self):
        return pylcp.quadrupoleMagneticField(self.alpha)

    # laser beams with detuning and saturation paramter.
    # geometry is infinite plane waves
    def laser_beams(self):
        return pylcp.conventional3DMOTBeams(delta=self.detuning,
                                            s=self.saturation,
                                            beam_type=pylcp.infinitePlaneWaveBeam)

    # gravity vector
    def gravity(self):
        gravity = -np.array([0., 0., 9.8 * self.time ** 2 / (self.length * 1e-2)])
        return gravity

    #
    def hamiltonian(self):
        hg, mug_q = pylcp.hamiltonians.singleF(F=0, muB=1)
        he, mue_q = pylcp.hamiltonians.singleF(F=1, muB=1)

        dq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)

        hamiltonian = pylcp.hamiltonian(hg, he,
                                        mug_q, mue_q, dq,
                                        mass=self.mass)
        return hamiltonian
