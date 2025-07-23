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
from parameters import linewidth, rabi_f, wg, we, N_max, N_i, mass, lamb, thetas, d_theta
from plot_utils import Plotting

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# simulation parameters
max_time_s = 1*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times_rabi = np.arange(0, max_time_rabi, dt)

# calculate solid state solution as a function of following detunings, rabi freqs.
num_detunings_ss = 10
num_rabifreqs_ss = 10
detunings_ss = 2*pi*np.linspace(-1*MHz, 0.2*MHz, num_detunings_ss)
rabi_freqs_ss = 2*pi*np.linspace(5*kHz, 355*kHz, num_rabifreqs_ss)

# prepare simulation
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)
Plot = Plotting('output')

# %% 2D scan over detuning and rabi frequency

# Allocate 2D result array
final_n = np.zeros((rabi_freqs_ss.size, detunings_ss.size))

# Prepare system
SisCooling = SisyphusCooling(N_max, N_i, mass, lamb, wg, thetas, d_theta)
psi0, project_e, project_g, number_op = SisCooling.get_operators()

print("Running 2D steadystate calculation over detuning and Rabi frequency...")

for i, rabi in enumerate(tqdm(rabi_freqs_ss, desc="Rabi scan")):
    for j, det in enumerate(detunings_ss):
        arguments = [linewidth, rabi, wg, we, det]
        H = SisCooling.calculate_H(*arguments)
        c_ops = SisCooling.calculate_c_ops(linewidth, rabi, thetas, d_theta)
        ss = steadystate(H, c_ops, method='direct')
        final_n[i, j] = expect(number_op, ss)

# %% plotting 2d scan result

# Convert to MHz and kHz for plotting axes
rabi_axis = rabi_freqs_ss/(2*pi*kHz)  # in kHz
detuning_axis = detunings_ss/(2*pi*kHz)  # in kHz

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(final_n, aspect='auto', origin='lower', cmap='bwr',
    extent=[detuning_axis[0], detuning_axis[-1], rabi_axis[0], rabi_axis[-1]])
cbar = plt.colorbar(im, ax=ax)
cbar.set_label(r"Final $\bar{n}$", rotation=270, labelpad=15)
ax.set_xlabel(r"Detuning $\Delta'/2\pi$ [kHz]")
ax.set_ylabel(r"Rabi frequency $\Omega/2\pi$ [kHz]")
ax.tick_params(axis="both", direction="in")
ax.grid(False)

#%% print ideal settings

# Find the minimum ⟨n⟩ and corresponding indices
min_n = np.min(final_n)
min_idx = np.unravel_index(np.argmin(final_n), final_n.shape)
i_opt, j_opt = min_idx

# Get optimal parameters
rabi_opt = rabi_freqs_ss[i_opt]         # in rad/s
detuning_opt = detunings_ss[j_opt]      # in rad/s

print(f"Minimum ⟨n⟩ = {min_n:.3f}")
print(f"Optimal Rabi frequency: {rabi_opt/(2*pi*kHz):.3f} MHz")
print(f"Optimal Detuning: {detuning_opt/(2*pi*kHz):.2f} kHz")

# %% print 1d slices of optimum point

# 1D slice at optimal detuning (i.e. vary Rabi frequency)
slice_vs_rabi = final_n[:, j_opt]  # fixed detuning (column)

fig_rabi, ax_rabi = plt.subplots(figsize=(4, 3))
ax_rabi.plot(rabi_freqs_ss/(2*pi*kHz), slice_vs_rabi, label=rf"$\Delta$' = {detuning_opt/(2*pi*kHz):.1f} kHz")
ax_rabi.set_xlabel(r'Rabi frequency $\Omega/2\pi$ [MHz]')
ax_rabi.set_ylabel(r'Final $\bar{n}$')
ax_rabi.grid()
ax_rabi.legend()
Plot.savefig("final_n_vs_rabi.pdf")

# 1D slice at optimal rabi (i.e. vary detuning)
slice_vs_det = final_n[i_opt, :]  # fixed rabi frequency (row)

fig_det, ax_det = plt.subplots(figsize=(4, 3))
ax_det.plot(detunings_ss/(2*pi*kHz), slice_vs_det, label=rf"$\Omega$' = {rabi_opt/(2*pi*kHz):.2f} kHz")
ax_det.set_xlabel(r"Detuning $\Delta' / 2\pi$ [kHz]")
ax_det.set_ylabel(r'Final $\bar{n}$')
ax_det.grid()
ax_det.legend()
Plot.savefig("final_n_vs_det.pdf")

#%%
