# this is the script from R.M.P. Teunissen on Sisyphus cooling, as described in:
    # Teunissen, R.M.P. (2023). Building Arrays of Individually Trapped 88Sr Atoms (MSc thesis). 
    # Eindhoven University of Technology, Eindhoven, The Netherlands.

# changes made: JIT compilation, MEsolve instead of MCsolve and some minor changes
    
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
from units import kHz, nm, ms, MHz
from sisyphus_cooling_class import SisyphusCooling
from parameters import linewidth, rabi_f, wg, we, detuning, mass, lamb, thetas, d_theta

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
N_max = 12      # motional levels
N_i = 3           # initial Fock level
max_time_s = 2*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg)

psi0, project_e, project_g, number_op = SisCooling.get_operators()
H = SisCooling.calculate_H(linewidth, rabi_f, wg, we, detuning)
c_ops = SisCooling.calculate_c_ops(linewidth, rabi_f, thetas, d_theta)

print("Running Sisyphus cooling simulation...")
sol = mesolve(H, psi0, times_rabi, c_ops, e_ops=[project_e, number_op], options={"store_states": False})

fig, ax = plt.subplots(figsize=(4, 3))
times_ms = times_rabi/rabi_f/ms # same rescaling as qubit mesolve expects, consistently scaled

ax.plot(times_ms, sol.expect[1], label=r"$\langle n \rangle$")
#plt.plot(times, res.expect[0], label=r"$P_e$")

ax.set_xlabel('Time [ms]')
ax.legend()
fig.tight_layout()

# Plot final n as a function of detuning
detunings_ss = 2*pi*np.linspace(-1.5*MHz, 1*MHz, 50)
final_motional_levels = np.zeros(detunings_ss.size)
base_freqs = [linewidth, rabi_f, wg, we, detuning]

print("Running steadystate calculation for different detunings...")
for i, det in enumerate(tqdm(detunings_ss)):
    # Update the detuning value for this iteration
    base_freqs[4] = det
    
    H = SisCooling.calculate_H(*base_freqs)
    c_ops = SisCooling.calculate_c_ops(linewidth, rabi_f, thetas, d_theta)
    ss = steadystate(H, c_ops, method='direct')

    # Calculate the expectation value for the motional number operator
    final_n = expect(number_op, ss)
    final_motional_levels[i] = final_n

fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.plot(detunings_ss/(2*pi*kHz), final_motional_levels)
ax2.set_xlabel(r"$\Delta/2\pi$ [kHz]")
ax2.set_ylabel(r"$\bar{n}$")
ax2.tick_params(axis="both",direction="in")
ax2.grid()

min_n = np.min(final_motional_levels)
detuning_min_n = detunings_ss[np.argmin(final_motional_levels)]
print(f"Lowest n = {min_n:.2f} found for a detuning of {detuning_min_n/(2*pi*1e3):.2f} kHz")
plt.show()

# # %% investigate cooling rate as a function of detuning

# # Redefine time to only take 0.1 ms  
# cooling_interval_s = 0.1*ms
# dt = 0.1 # [Rabi cycles]
# cooling_interval_rabi = cooling_interval_s*rabi_f # time in Rabi cycles. 
# coolig_times_rabi = np.arange(0, cooling_interval_rabi, dt)

# detunings = 2*pi*np.linspace(-400*kHz, 0*kHz, 25)
# final_ns_vs_det = np.zeros(detunings.size)
# base_freqs = base_freqs.copy()

# print("Running Sisyphus cooling simulation as a function of detuning. ...")

# for i, detuning in enumerate(detunings):
#     print(f"run: {i + 1}/{final_ns_vs_det.size}")
#     base_freqs[4] = detuning
#     sol = solve_master_equation(base_freqs, coolig_times_rabi)
#     n_final = sol.expect[1][-1]
#     final_ns_vs_det[i] = n_final

# n_reductions_vs_det = N_i*np.ones(detunings.size) - final_ns_vs_det

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.plot(detunings/(2*pi)/kHz, n_reductions_vs_det, label=r"$\Omega/2\pi$ ="+ f"{rabi_f/(2*pi*1e3):.0f} kHz")
# ax.set_xlabel(r"Detuning $\Delta/2\pi$ [kHz]")
# ax.set_ylabel(r"$\bar{n}_{f} - \bar{n}_{i}$")
# ax.legend()
# ax.tick_params(axis="both", direction="in")

# # %% Plot the cooling rate as a function of Rabi freq. 

# rabi_freqs = 2*pi*np.linspace(20, 200, 25)*kHz
# final_ns_vs_rabi = np.zeros(rabi_freqs.size)

# print("Running Sisyphus cooling simulation as a function of Rabi freq. ...")

# for i, new_rabi_f in enumerate(tqdm(rabi_freqs)):
#     print(f"run: {i + 1}/{final_ns_vs_rabi.size}")
#     base_freqs[1] = new_rabi_f
#     sol = solve_master_equation(base_freqs, coolig_times_rabi)
#     n_final = sol.expect[1][-1]
#     final_ns_vs_rabi[i] = n_final

# n_reductions_vs_rabi = N_i*np.ones(rabi_freqs.size) - final_ns_vs_rabi

# fig, ax = plt.subplots(figsize=(4, 3))
# ax.plot(rabi_freqs/(2*pi)/kHz, n_reductions_vs_rabi, label=r"$\Delta/2\pi$ ="+ f"{detuning/(2*pi*1e3):.0f} kHz")
# ax.set_xlabel(r"Rabi frequency $\Omega/2\pi$ [kHz]")
# ax.set_ylabel(r"$\bar{n}_{f} - \bar{n}_{i}$")
# ax.tick_params(axis="both", direction="in")
# plt.legend()
# plt.grid(linewidth = 0.5, linestyle = "-.")
# plt.show()

# # %%
