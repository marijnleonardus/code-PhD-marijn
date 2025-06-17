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
from units import kHz, ms, MHz
from sisyphus_cooling_class import SisyphusCooling
from parameters import linewidth, rabi_f, wg, we, detuning, mass, lamb, thetas, d_theta
from plotting_class import Plotting

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
N_max = 10      # motional levels
N_i = 3           # initial Fock level
max_time_s = 1*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# calculate solid state solution as a function of following detunings
num_detunings_ss = 6
detunings_ss = 2*pi*np.linspace(-1.5*MHz, 1*MHz, num_detunings_ss)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)

# %% 
print("Running Sisyphus cooling simulation...")
sol = SisCooling.solve_master_equation([linewidth, rabi_f, we, detuning, times_rabi])

fig, ax = plt.subplots(figsize=(4, 3))
times_ms = times_rabi/rabi_f/ms # same rescaling as qubit mesolve expects, consistently scaled

ax.plot(times_ms, sol.expect[1])
#plt.plot(times, res.expect[0], label=r"$P_e$")
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r"$\bar{n}$")
fig.tight_layout()
Plotting.savefig('output', 'sis_cooling_time_evolution.pdf')

# %% 
# Plot final n as a function of detuning
final_motional_levels = np.zeros(detunings_ss.size)
arguments = [linewidth, rabi_f, wg, we, detuning]

psi0, project_e, project_g, number_op = SisCooling.get_operators()

print("Running steadystate calculation for different detunings...")
for i, det in enumerate(tqdm(detunings_ss)):
    # we don't use the function'solve_master_equation here, but rather calculate the steady state directly
    # because we are interested in the steady state for different detunings

    # Update the detuning value for this iteration
    arguments[4] = det
    
    H = SisCooling.calculate_H(*arguments)
    c_ops = SisCooling.calculate_c_ops(linewidth, rabi_f, thetas, d_theta)
    ss = steadystate(H, c_ops, method='direct')

    # Calculate the expectation value for the motional number operator
    final_n = expect(number_op, ss)
    final_motional_levels[i] = final_n

# %% 
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.plot(detunings_ss/(2*pi*kHz), final_motional_levels)
ax2.set_xlabel(r"$\Delta'/2\pi$ [kHz]")
ax2.set_ylabel(r"$\bar{n}$")
ax2.tick_params(axis="both",direction="in")
ax2.grid()

min_n = np.min(final_motional_levels)
detuning_min_n = detunings_ss[np.argmin(final_motional_levels)]
print(f"Lowest n = {min_n:.2f} found for a detuning of {detuning_min_n/(2*pi*1e3):.2f} kHz")
Plotting.savefig('output', 'sis_cooling_steady_state.pdf')

# %%
