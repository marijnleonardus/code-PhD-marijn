from pylcp_class import RecoilLimitedMOT
import numpy as np
import scipy.constants as cts
from scipy.constants import pi, hbar
import pylcp
import matplotlib.pyplot as plt
import pathos
from matplotlib.patches import Ellipse


def main():
    # independent variables
    wavenumber = 2 * pi / 689e-7  # 1/cm
    linewidth = 2 * pi * 7.4e3  # 1/s, thus kHz linewidth
    detuning = -200/7.5
    saturation = 25
    bohr_magneton = cts.value('Bohr magneton in Hz/T')
    atomic_mass_unit = cts.value('atomic mass constant')

    # dependent variables
    length = 1/wavenumber  # cm
    time = 1/linewidth  # s
    alpha = 1.5 * bohr_magneton * 1e-4 * 8 * length / (linewidth / (2 * pi))
    mass = 87.8 * atomic_mass_unit * (length * 1e-2)**2 / hbar / time

    # create object
    sr_red_mot = RecoilLimitedMOT(detuning, saturation, length, time, alpha, mass)

    govern_eqn = pylcp.rateeq(sr_red_mot.laser_beams(),
                              sr_red_mot.magnetic_field(),
                              sr_red_mot.hamiltonian(),
                              sr_red_mot.gravity()
                              )

    # dynamics
    max_time = 0.05 / time
    if isinstance(govern_eqn,
                  pylcp.rateeq):
        govern_eqn.set_initial_pop(np.array([1., 0., 0., 0.]))
    govern_eqn.set_initial_position(np.array([0., 0., 0.]))

    govern_eqn.evolve_motion([0, 0.05 / time],
                             random_recoil=True,
                             progress_bar=True,
                             max_step=1.)

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
    ax[0].plot(govern_eqn.sol.t * time, govern_eqn.sol.r.T * (1e4 * length))
    ax[1].plot(govern_eqn.sol.t * time, govern_eqn.sol.v.T)
    ax[0].set_ylabel('$r$ ($\mu$m)')
    ax[0].set_xlabel('$t$ (s)')
    ax[1].set_ylabel('$v/(\Gamma/k)$')
    ax[1].set_xlabel('$t$ (s)')
    fig.subplots_adjust(left=0.08, wspace=0.22)
    plt.show()


if __name__ == '__main__':
    main()
