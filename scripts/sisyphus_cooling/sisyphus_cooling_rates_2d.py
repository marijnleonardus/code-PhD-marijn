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
import os

# Add lib/ to sys.path so we can import setup_paths
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

from units import kHz, ms, MHz
from sisyphus_cooling_class import SisyphusCooling
from plot_utils import Plotting
from parameters import linewidth, rabi_f, wg, we, N_i, N_max, mass, lamb, thetas, d_theta

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
max_time_s = 0.5*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

num_rabi_freqs = 5
num_detunings = 5
detunings_array = 2*pi*np.linspace(-0.4*MHz, 0.2*MHz, num_detunings)
rabifreqs_array = 2*pi*np.linspace(5*kHz, 355*kHz, num_rabi_freqs)

# %% cooling rate: 2D scan over detuning and rabi frequency

# Allocate 2D result array
final_n = np.zeros((num_rabi_freqs, num_detunings))

# Prepare system
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)
psi0, project_e, project_g, number_op = SisCooling.get_operators()

print("Running 2D cooling rate calculation over detuning and Rabi frequency...")

# 4) Loop over both dimensions
for i, Ω in enumerate(tqdm(rabifreqs_array, desc="Ω sweep")):
    for j, Δ in enumerate(tqdm(detunings_array, desc="Δ sweep", leave=False)):
        # update arguments: [linewidth, rabi_freq, we, detuning, times_rabi]
        args = [linewidth, Ω, we, Δ, times_rabi]
        sol = SisCooling.solve_master_equation(args)
        n_final = sol.expect[1][-1]
        final_n[i, j] = n_final

# %% 

np.savetxt('sims/sol.txt', sol)

# %% 
n_change = final_n - N_i*np.ones((num_rabi_freqs, num_detunings)) 
dndt = n_change/max_time_s

# Convert to MHz and kHz for plotting axes
rabi_axis = rabifreqs_array/(2*pi*kHz)  # in kHz
detuning_axis = detunings_array/(2*pi*kHz)  # in kHz

Plot = Plotting('output')

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(dndt/(1/ms), aspect='auto', origin='lower', cmap='bwr',
    extent=[detuning_axis[0], detuning_axis[-1], rabi_axis[0], rabi_axis[-1]])
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"Cooling rate $\Delta n/\Delta t$ [ms$^{-1}$]", rotation=270, labelpad=15)
ax.set_xlabel(r"Detuning $\Delta'/2\pi$ [kHz]")
ax.set_ylabel(r"Rabi frequency $\Omega/2\pi$ [kHz]")
ax.tick_params(axis="both", direction="in")
ax.grid(False)
Plot.savefig('coolingrate_2d.pdf')

#%% print ideal settings

# Find the minimum ⟨n⟩ and corresponding indices
max_dndt = np.max(dndt)
max_dndt_idx = np.unravel_index(np.argmax(dndt), dndt.shape)
i_opt, j_opt = max_dndt_idx

# Get optimal parameters
rabi_opt = rabifreqs_array[i_opt]         # in rad/s
detuning_opt = detunings_array[j_opt]      # in rad/s

print(f"max d⟨n⟩/dt = {max_dndt/(1/ms):.3f}")
print(f"Optimal Rabi frequency: {rabi_opt/(2*pi*kHz):.3f} kHz")
print(f"Optimal Detuning: {detuning_opt/(2*pi*kHz):.2f} kHz")

# %% print 1d slices of optimum point

# 1D slice at optimal detuning (i.e. vary Rabi frequency)
slice_vs_rabi = dndt[:, j_opt]  # fixed detuning (column)

fig_rabi, ax_rabi = plt.subplots(figsize=(4, 3))
ax_rabi.plot(rabifreqs_array/(2*pi*kHz), slice_vs_rabi/(1/ms), 
    label=rf"$\Delta'/2\pi$ = {detuning_opt/(2*pi*kHz):.1f} kHz")
ax_rabi.set_xlabel(r'Rabi frequency $\Omega/2\pi$ [MHz]')
ax_rabi.set_ylabel(r"Cooling rate $\Delta n/\Delta t$ [ms$^{-1}$]")
ax_rabi.grid()
ax_rabi.legend()
Plot.savefig("cooling_rate_vs_rabi.pdf")

# 1D slice at optimal rabi (i.e. vary detuning)
slice_vs_det = dndt[i_opt, :]  # fixed rabi frequency (row)

fig_det, ax_det = plt.subplots(figsize=(4, 3))
ax_det.plot(detunings_array/(2*pi*kHz), slice_vs_det/(1/ms), 
    label=rf"$\Omega/2\pi =$ {rabi_opt/(2*pi*kHz):.2f} kHz")
ax_det.set_xlabel(r"Detuning $\Delta' / 2\pi$ [kHz]")
ax_det.set_ylabel(r"$\Delta n/\Delta t$ [ms$^{-1}$]")
ax_det.grid()
ax_det.legend()
Plot.savefig("cooling_rate_vs_detuning.pdf")

# %%

plt.show()