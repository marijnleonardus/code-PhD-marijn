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
from parameters import linewidth, rabi_f, wg, we, detuning, mass, lamb, thetas, d_theta
from plot_utils import Plotting

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
N_max = 20      # motional levels
N_i = 12           # initial Fock level
max_time_s = 5*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# calculate solid state solution as a function of following detunings
num_detunings_ss = 86
detunings_ss = 2*pi*np.linspace(-1.5*MHz, 0.2*MHz, num_detunings_ss)

# calculate solid state sol. for rabi freqs.
num_rabifreqs_ss = 101
rabi_freqs_ss = 2*pi*np.linspace(5*kHz, 355*kHz, num_rabifreqs_ss)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)

# %% 
print("Running Sisyphus cooling simulation...")
sol = SisCooling.solve_master_equation([linewidth, rabi_f, we, detuning, times_rabi])

avg_n = sol.expect[1]
pop_g_0 = sol.expect[2]
pop_e_0 = sol.expect[3]
pop_ground = pop_g_0 + pop_e_0
times_ms = times_rabi/rabi_f/ms # same rescaling as qubit mesolve expects, consistently scaled

# %% 
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

# %% Plot final n as a function of detuning

arguments = [linewidth, rabi_f, wg, we, detuning]
psi0, project_e, project_g, number_op = SisCooling.get_operators()

print("Running steadystate calculation for different detunings...")

final_n_det = np.zeros(detunings_ss.size)
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
    final_n_det[i] = final_n

# %% 
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.plot(detunings_ss/(2*pi*kHz), final_n_det, label=r"$\Omega/2\pi$ ="+ f"{rabi_f/(2*pi*1e3):.0f} kHz")
ax2.set_xlabel(r"$\Delta'/2\pi$ [kHz]")
ax2.set_ylabel(r"$\bar{n}$")
ax2.tick_params(axis="both", direction="in")
ax2.legend()
ax2.grid()

min_n_det = np.min(final_n_det)
detuning_min_n = detunings_ss[np.argmin(final_n_det)]
print(f"Lowest n = {min_n_det:.2f} found for a detuning of {detuning_min_n/(2*pi)/kHz:.2f} kHz")
Plot.savefig('sis_cooling_ss_det.pdf')

# %% plot final n as a function of rabi freq. 

arguments = [linewidth, rabi_f, wg, we, detuning]
psi0, project_e, project_g, number_op = SisCooling.get_operators()

print("Running steadystate calculation for different rabi frequencies...")

final_n_rabi = np.zeros(rabi_freqs_ss.size)
for i, rabi in enumerate(tqdm(rabi_freqs_ss)):
    # we don't use the function'solve_master_equation here, but rather calculate the steady state directly
    # because we are interested in the steady state for different detunings

    # Update the detuning value for this iteration
    arguments[1] = rabi
    
    H = SisCooling.calculate_H(*arguments)
    c_ops = SisCooling.calculate_c_ops(linewidth, rabi_f, thetas, d_theta)
    ss = steadystate(H, c_ops, method='direct')

    # Calculate the expectation value for the motional number operator
    final_n = expect(number_op, ss)
    final_n_rabi[i] = final_n

# %% 

fig3, ax3 = plt.subplots(figsize=(4, 3))
ax3.plot(rabi_freqs_ss/(2*pi*kHz), final_n_rabi, label=r"$\Delta/2\pi$ ="+ f"{detuning/(2*pi*kHz):.0f} kHz")
ax3.set_xlabel(r"$\Omega/2\pi$ [kHz]")
ax3.set_ylabel(r"$\bar{n}$")
ax3.tick_params(axis="both", direction="in")
ax3.legend()
ax3.grid()

min_n_rabi = np.min(final_n_rabi)
rabi_min_n = rabi_freqs_ss[np.argmin(final_n_rabi)]
print(f"Lowest n = {min_n_rabi:.2f} found for a rabi freq. of {rabi_min_n/(2*pi)/kHz:.2f} kHz")
Plot.savefig('sis_cooling_ss_rabi.pdf')

# %%
