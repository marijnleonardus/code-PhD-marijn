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
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

from units import kHz, ms, MHz
from sisyphus_cooling_class import SisyphusCooling
from parameters import linewidth, rabi_f, wg, we, N_i, N_max, detuning, mass, lamb, thetas, d_theta
from plot_utils import Plotting

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

max_time_s = 5*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)

# %% run sis simulation as a function of time

print("Running Sisyphus cooling simulation...")
sol = SisCooling.solve_master_equation([linewidth, rabi_f, we, detuning, times_rabi])

avg_n = sol.expect[1]
pop_g_0 = sol.expect[2]
pop_e_0 = sol.expect[3]
pop_ground = pop_g_0 + pop_e_0
times_ms = times_rabi/rabi_f/ms # same rescaling as qubit mesolve expects, consistently scaled

# %% plot time dependent result

Plot = Plotting('output')
fig, ax = plt.subplots(figsize=(4, 3))
ax.grid()
ax.plot(times_ms, avg_n)
ax.set_xlabel('Time [ms]')
ax.set_yscale('log')
ax.set_ylabel(r"$\bar{n}$")
Plot.savefig('sis_cooling_time_evolution.pdf')

fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.grid()
ax2.plot(times_ms, pop_g_0, label=r'$P($g$, n=0)$')
ax2.plot(times_ms, pop_e_0, label=r'$P($e$, n=0)$')
ax2.plot(times_ms, pop_ground, label=r'$P($g$+$e$, n=0)$')
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('Population')
ax2.legend()
Plot.savefig('sis_cooling_populations.pdf')

# %%
