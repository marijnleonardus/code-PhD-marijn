# author: Marijn Venderbosch
# october 2022

# %% imports

from pylcp_class import RecoilLimitedMOT
import numpy as np
import scipy.constants as cts
from scipy.constants import pi, hbar
import pylcp
import matplotlib.pyplot as plt
#import pathos
#from matplotlib.patches import Ellipse

# %% variables

# red transition paramters 
wavelength = 689e-9 *1e2  # cm
linewidth = 2 * pi * 7.4e3  # rad/s, thus kHz linewidth

# laser cooling settings
detuning = -200/7.5  # dimensionless
saturation = 25   # ,,

# constants
bohr_magneton = cts.value('Bohr magneton in Hz/T')  # Hz/T
atomic_mass_unit = cts.value('atomic mass constant')  # kg

# characteristic length/time scales
wavenumber = 2 * pi / wavelength  # 1/cm
length = 1 / wavenumber  # cm
time = 1 / linewidth  # s

# dependent variables
alpha = 1.5 * bohr_magneton * 1e-4 * 8 * length / (linewidth / (2 * pi))  
mass = 87.8 * atomic_mass_unit * (length * 1e-2)**2 / hbar / time


# %% functions

def main():
    
    # create object
    sr_red_mot = RecoilLimitedMOT(detuning, saturation, length, time, alpha, mass)
    
    # define rate equation
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

    fig, (ax0, ax1) = plt.subplots(1, 2,
                                   figsize=(6.5, 2.75))
    fig.subplots_adjust(left=0.08, wspace=0.22)
    
    ax0.plot(govern_eqn.sol.t * time, govern_eqn.sol.r.T * (1e4 * length))
    ax0.set_xlabel('$t$ (s)')
    ax0.set_ylabel('$r$ ($\mu$m)')
    
    ax1.plot(govern_eqn.sol.t * time, govern_eqn.sol.v.T)
    ax1.set_xlabel('$t$ (s)')
    ax1.set_ylabel('$v/(\Gamma/k)$')

    plt.show()


if __name__ == '__main__':
    main()
