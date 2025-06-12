# this is the script from R.M.P. Teunissen on Sisyphus cooling, as described in:
    # Teunissen, R.M.P. (2023). Building Arrays of Individually Trapped 88Sr Atoms (MSc thesis). 
    # Eindhoven University of Technology, Eindhoven, The Netherlands.

# changes made: JIT compilation, MEsolve instead of MCsolve and some minor changes

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
from units import kHz, nm, ms, MHz
from sisyphus_cooling_class import SisyphusCooling
from parameters import linewidth, rabi_f, wg, we, detuning, mass, lamb, thetas, d_theta

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
N_max = 10      # motional levels
N_i = 3           # initial Fock level
max_time_s = 0.2*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# %% Plot the cooling rate as a function of Rabi freq. 

rabi_freqs = 2*pi*np.linspace(20, 200, 5)*kHz
final_ns_vs_rabi = np.zeros(rabi_freqs.size)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg)
psi0, project_e, project_g, number_op = SisCooling.get_operators()


def solve_master_equation(freqs, times_rabi):
    linewidth, rabi_f, wg, we, detuning = freqs
    H = SisCooling.calculate_H(linewidth, rabi_f, wg, we, detuning)
    c_ops = SisCooling.calculate_c_ops(linewidth, rabi_f, thetas, d_theta)
    sol = mesolve(H, psi0, times_rabi, c_ops, e_ops=[project_e, number_op], options={"store_states": False})
    return sol


print("Running Sisyphus cooling simulation as a function of Rabi freq. ...")
freqs = [linewidth, rabi_f, wg, we, detuning]

for i, new_rabi_f in enumerate(tqdm(rabi_freqs)):
    print(f"run: {i + 1}/{final_ns_vs_rabi.size}")
    freqs[1] = new_rabi_f
    sol = solve_master_equation(freqs, times_rabi)
    n_final = sol.expect[1][-1]
    final_ns_vs_rabi[i] = n_final

n_reductions_vs_rabi = N_i*np.ones(rabi_freqs.size) - final_ns_vs_rabi

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(rabi_freqs/(2*pi)/kHz, n_reductions_vs_rabi, label=r"$\Delta/2\pi$ ="+ f"{detuning/(2*pi*1e3):.0f} kHz")
ax.set_xlabel(r"Rabi frequency $\Omega/2\pi$ [kHz]")
ax.set_ylabel(r"$\bar{n}_{f} - \bar{n}_{i}$")
ax.tick_params(axis="both", direction="in")
plt.legend()
plt.grid(linewidth = 0.5, linestyle = "-.")
plt.show()

# %%
