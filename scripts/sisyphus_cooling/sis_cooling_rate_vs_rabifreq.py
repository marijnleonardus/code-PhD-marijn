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
from plotting_class import Plotting

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
N_max = 10      # motional levels
N_i = 3           # initial Fock level
time_interval = 0.05*ms
dt = 0.1
max_time_rabi = time_interval*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)
num_rabi_frequencies_sim = 3

# %% Plot the cooling rate as a function of Rabi freq. 

rabi_freqs = 2*pi*np.linspace(5, 250, num_rabi_frequencies_sim)*kHz
final_ns_vs_rabi = np.zeros(rabi_freqs.size)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)

print("Running Sisyphus cooling simulation as a function of Rabi freq. ...")
arguments = [linewidth, rabi_f, we, detuning, times_rabi]

for i, new_rabi_f in enumerate(tqdm(rabi_freqs)):
    print(f"run: {i + 1}/{final_ns_vs_rabi.size}")
    arguments[1] = new_rabi_f
    sol = SisCooling.solve_master_equation(arguments)
    n_final = sol.expect[1][-1]
    final_ns_vs_rabi[i] = n_final

# %% 

n_reductions_vs_rabi = N_i*np.ones(rabi_freqs.size) - final_ns_vs_rabi
cooling_rate = n_reductions_vs_rabi/time_interval  # delta n / ms

fig, ax = plt.subplots(figsize=(4, 3))
ax.grid()
ax.plot(rabi_freqs/(2*pi)/kHz, cooling_rate/(1/ms), label=r"$\Delta/2\pi$ ="+ f"{detuning/(2*pi*kHz):.0f} kHz")
ax.set_xlabel(r"Rabi frequency $\Omega/2\pi$ [kHz]")
ax.set_ylabel(r"Cooling rate $\Delta n/\Delta t$ [ms$^{-1}$]")
ax.tick_params(axis="both", direction="in")
plt.legend()
Plotting.savefig('output', 'sis_cooling_rate_vs_rabifreq.pdf')

# %%
