# %%
# this is the script from R.M.P. Teunissen on Sisyphus cooling, as described in:
    # Teunissen, R.M.P. (2023). Building Arrays of Individually Trapped 88Sr Atoms (MSc thesis). 
    # Eindhoven University of Technology, Eindhoven, The Netherlands.

# changes made: JIT compilation, MEsolve instead of MCsolve and some minor changes
    

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import proton_mass, hbar, pi
from qutip import *

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
from units import kHz, nm, ms, MHz
from atoms_tweezer_class import AtomicMotion

# QuTiP settings for performance
# enable automatic cleanup of negligible elements
qutip.settings.auto_tidyup = True
qutip.settings.auto_tidyup_atol = 1e-12

# %% parameters

# % physical parameters
linewidth = 2*pi*7.4*kHz
rabi_f = 2*pi*100*kHz
wg = 2*pi*86*kHz
alpha_e = 355 # atomic polarizability in atomic units
alpha_g = 286 # atomic polarizability in atomic units
we = np.sqrt(alpha_e/alpha_g)*wg
detuning = -1.7*wg
mass = 87.9*proton_mass
lamb = 689*nm

# simulation parameters
N_max = 15        # motional levels
N_i = 5           # initial Fock level
max_time_s = 10*ms
dt = 0.1
max_time_rabi = max_time_s*rabi_f # time in Rabi cycles. 
# Confusing, but QuTip mesolve expects time in Rabi cycles
# as t_nondimensionalized = t*real*omega_ref
times = np.arange(0, max_time_rabi, dt)

# emission angle discretization
N_theta = 10
thetas = np.linspace(0, pi, N_theta)
d_theta = thetas[1] - thetas[0]

# system operators
# two-level atom tensor harmonic oscillator
psi0 = tensor(basis(2, 0), fock(N_max, N_i))
a = tensor(qeye(2), destroy(N_max))
emit = tensor(Qobj([[0, 1], [0, 0]]), qeye(N_max))
absorb = emit.dag()
project_e = tensor(projection(2, 1, 1), qeye(N_max))
project_g = tensor(projection(2, 0, 0), qeye(N_max))
number_op = a.dag()*a

# %% helper functions


def rad_pattern(theta, d_theta, me=1):
    """Radiation pattern integration weight"""
    if me == 0:
        P = 3/4*np.sin(theta)**2
    else:
        P = 3/8*(1 + np.cos(theta)**2)
    return P*np.sin(theta)*d_theta


def calculate_H(linewidth, rabi_f, wg, we, detuning):
    """Compute sparse Hamiltonian (nondimensional units)"""

    # compute Lamb-Dicke parameter
    eta = AtomicMotion().lamb_dicke_parameter(mass, lamb, wg)

    # nondimensionalize
    freqs = np.array([linewidth, rabi_f, wg, we, detuning])/rabi_f
    lw_nd, rabi_nd, wg_nd, we_nd, det_nd = freqs

    H = (
        wg_nd*(a.dag()*a+0.5)*(project_g+project_e)
        + (we_nd**2-wg_nd**2)/(4*wg_nd)*(a+a.dag())**2*project_e
        + rabi_nd/2*(
            (1j*eta*(a+a.dag())).expm()*absorb
            + (-1j*eta*(a+a.dag())).expm()*emit
        ) - det_nd*project_e)
    return H


def calculate_c_ops(linewidth, rabi_f, thetas, d_theta):
    """Precompute sparse collapse operators for given angles"""
    
    linewidth_nd = linewidth/rabi_f
    c_ops = []
    for theta in thetas:
        rate = linewidth_nd*rad_pattern(theta, d_theta)
        kick = (-1j*(a + a.dag())*np.cos(theta)).expm()
        c_ops.append(np.sqrt(rate)*kick*emit)
    return c_ops


# %% do one simulation for particular settings used

print("Running Sisyphus cooling simulation...")

H = calculate_H(linewidth, rabi_f, wg, we, detuning)
c_ops = calculate_c_ops(linewidth, rabi_f, thetas, d_theta)
res = mesolve(H, psi0, times, c_ops, e_ops=[project_e, number_op], options={"store_states": False})

# %% plot results

plt.figure()
#plt.plot(times, res.expect[0], label=r"$P_e$")
times_ms = times/rabi_f/ms # same rescaling as qubit mesolve expects, consistently scaled
plt.plot(times_ms, res.expect[1], label=r"$\langle n \rangle$")
plt.xlabel('Time [ms]')
plt.legend()
plt.tight_layout()
plt.show()

# %% solve as a function of detuning

detunings = 2*pi*np.linspace(-1.5*MHz, 1*MHz, 30)
base_freqs = [linewidth, rabi_f, wg, we, detuning]

final_motional_levels = np.zeros(detunings.size)

print("Running steadystate calculation for different detunings...")
for i, d in enumerate(tqdm(detunings)):
    # Update the detuning value for this iteration
    base_freqs[4] = d
    
    # Calculate the Hamiltonian by UNPACKING the list of frequencies
    H = calculate_H(*base_freqs)
    
    # Find the steady state
    ss = steadystate(H, c_ops, method='direct') # Using a specific method can sometimes help

    # Calculate the expectation value for the motional number operator
    final_n = expect(number_op, ss)
    final_motional_levels[i] = final_n

# %% plot the final motional levels as a function of detuning

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(detunings/(2*pi*kHz), final_motional_levels)
ax.set_xlabel(r"$\Delta/2\pi$ [kHz]")
ax.set_ylabel(r"$\left<n\right>_{end}$")
ax.tick_params(axis="both",direction="in")
ax.grid(linewidth = 0.5, linestyle = "-.")
plt.show()

min_n = np.min(final_motional_levels)
detuning_min_n = detunings[np.argmin(final_motional_levels)]
print(f"Lowest motional level of n = {min_n:.2f} was found for a detuning of {detuning_min_n/(2*pi*1e3):.2f} kHz")

# %%
