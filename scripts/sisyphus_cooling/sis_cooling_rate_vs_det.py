# adapted from the script from R.M.P. Teunissen on Sisyphus cooling, as described in:
    # Teunissen, R.M.P. (2023). Building Arrays of Individually Trapped 88Sr Atoms (MSc thesis). 
    # Eindhoven University of Technology, Eindhoven, The Netherlands.

# refactored code in class and seperate main functinos, MEsolve instead of MCsolve and some minor changes

# %%     
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import pi
from qutip import *

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
from units import kHz, ms
from sisyphus_cooling_class import SisyphusCooling
from parameters import linewidth, rabi_f, wg, we, detuning, mass, lamb, thetas, d_theta

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
N_max = 20      # motional levels
N_i = 5           # initial Fock level
time_interval = 0.2*ms
dt = 0.1
max_time_rabi = time_interval*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# %% calculate cooling rate as a function of detuning

detunings = 2*pi*np.linspace(-400*kHz, 0*kHz, 41)
final_ns_vs_det = np.zeros(detunings.size)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)

print("Running Sisyphus cooling simulation as a function of detuning. ...")
arguments = [linewidth, rabi_f, we, detuning, times_rabi]

for i, detuning in enumerate(detunings):
    print(f"run: {i + 1}/{final_ns_vs_det.size}")
    arguments[3] = detuning
    sol = SisCooling.solve_master_equation(arguments)
    n_final = sol.expect[1][-1]
    final_ns_vs_det[i] = n_final

n_reductions_vs_det = N_i*np.ones(detunings.size) - final_ns_vs_det

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(detunings/(2*pi)/kHz, n_reductions_vs_det, label=r"$\Omega/2\pi$ ="+ f"{rabi_f/(2*pi*1e3):.0f} kHz")
ax.set_xlabel(r"Detuning $\Delta/2\pi$ [kHz]")
ax.set_ylabel(r"$\bar{n}_{f} - \bar{n}_{i}$")
ax.legend()
ax.tick_params(axis="both", direction="in")

plt.show()
