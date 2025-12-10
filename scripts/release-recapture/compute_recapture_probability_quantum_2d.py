"""
Marijn Venderbosch 
November 2025

Code for computing recapture probability after turning off tweezers for variable duration.

MODIFICATION:
- Initialization is performed in the Harmonic Oscillator (HO) basis.
- Evolution is performed in free space.
- Recapture projection is performed onto the actual Gaussian bound states.
- Visualization of Gaussian Potential + Eigenstates included.
- ADDED: 2D Recapture Probability using (n + 1) degeneracy weighting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.constants import pi, Boltzmann, hbar, atomic_mass
import math
import time

# add raw data import
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(script_dir,'../../raw_data/release_recapture/') # adjust as needed
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
utils_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))

sys.path.append(modules_dir)
sys.path.append(utils_dir)
sys.path.append(data_dir)

# --- Unit conversions ---
from units import uK, um, us, kHz
from plot_utils import Plotting

# --- Parameters ---
trap_depth = 450*uK
trap_frequency = 82*kHz
#trap_depth = 190*uK*0.75/0.82 # the latter is a correction factor on the trap depth
temperature = 5*uK 
#waist_computed = 0.7*um
m = 88*atomic_mass
n_ho_basis = 150  
number_x_grid_points = int(2**15) 
max_radius = 8  # in HO units
max_release_time_s = 80*us
nr_tau_values = 30 
use_exp_data = False

# --- Derived parameters ---
#trap_frequency= 2*np.sqrt(Boltzmann*trap_depth/(m*waist_computed**2))/(2*pi)
print(f"Trap Frequency: {trap_frequency/kHz:.2f} kHz")

U0 = Boltzmann*trap_depth/(hbar*2*pi*trap_frequency) 
t_natural = Boltzmann*temperature/(hbar*2*pi*trap_frequency) 
trap_freq_rad = 2*pi*trap_frequency
tweezer_waist_m = 2*np.sqrt(Boltzmann*trap_depth/(m*trap_freq_rad**2))

# Grid setup
spread_factor = np.sqrt(1 + (1.0*max_release_time_s*trap_freq_rad)**2)
x_max = max(max_radius, max_radius*spread_factor) 
x_grid = np.linspace(-x_max, x_max, number_x_grid_points)


class OpticalTweezer:
    """
    Compute recapture probability for an atom in a Gaussian optical tweezer.
    """
    
    def __init__(self, U0, omega, x_grid, n_ho_basis=100):
        self.U0 = U0
        self.omega = omega
        self.x_grid = x_grid
        self.dx = x_grid[1] - x_grid[0]
        
        self.a_ho = 1.0/np.sqrt(omega)
        self.n_ho_basis = n_ho_basis
        self.waist = 2*self.a_ho*np.sqrt(U0/omega)
        
        self._norms = np.array([1.0/np.sqrt(2**n*math.factorial(n)*self.a_ho*np.sqrt(pi)) 
            for n in range(n_ho_basis)])
        
        print("Solving Gaussian eigenstates...")
        self.gauss_energies, self.gauss_eigenvectors = self._solve_gaussian_potential(x_grid)
        
        print("Pre-computing HO basis states...")
        self.ho_eigenvectors = self._harmonic_oscillator_wf_batch(x_grid)

    def _harmonic_oscillator_wf_batch(self, x):
        """Stable computation of HO wavefunctions"""
        x = x/self.a_ho
        basis_states = np.zeros((self.n_ho_basis, len(x)))
        psi_prev = self._norms[0]*np.exp(-x**2/2)
        basis_states[0] = psi_prev

        if self.n_ho_basis > 1:
            psi_curr = self._norms[1]*(2*x)*np.exp(-x**2/2)
            basis_states[1] = psi_curr
            for n in range(1, self.n_ho_basis - 1):
                psi_next = (np.sqrt(2/(n + 1))*x*psi_curr - np.sqrt(n/(n + 1))*psi_prev)
                basis_states[n + 1] = psi_next
                psi_prev, psi_curr = psi_curr, psi_next
        return basis_states
    
    def _gaussian_potential(self, x):
        return -self.U0*np.exp(-2*x**2/self.waist**2)
    
    def _solve_gaussian_potential(self, x_grid):
        H = np.diag([(n + 0.5)*self.omega for n in range(self.n_ho_basis)])
        psi_matrix = self._harmonic_oscillator_wf_batch(x_grid)
        V_perturbation = self._gaussian_potential(x_grid) - (0.5*self.omega**2*x_grid**2)
        
        weighted_psi = psi_matrix*V_perturbation[np.newaxis, :]
        V_matrix = np.dot(weighted_psi, psi_matrix.T)*self.dx
        H += V_matrix
        
        energies, eigenvectors = eigh(H)
        bound_mask = energies < 0
        return energies[bound_mask], eigenvectors[:, bound_mask]

    def get_gaussian_bound_states_spatial(self):
        coeffs = self.gauss_eigenvectors
        basis = self.ho_eigenvectors
        return np.dot(coeffs.T, basis)

    def free_evolution(self, psi_0, tau, x):
        dx = x[1] - x[0]
        psi_k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_0)))*dx/np.sqrt(2*pi)
        N = len(x)
        k = np.fft.fftshift(np.fft.fftfreq(N, dx/(2*pi)))
        phase = np.exp(-1j*k**2*tau/2)
        psi_k_evolved = psi_k*phase
        psi_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psi_k_evolved)))*np.sqrt(2*pi)/dx
        return psi_t

    def compute_overlap_batch(self, psi_evolved, bound_states_spatial):
        overlaps = np.dot(bound_states_spatial, psi_evolved)*self.dx
        return np.abs(overlaps)**2

    def thermal_recapture_scan_2D_weighted(self, tau_values, temp_natural):        
        """
        Computes 1D recapture and then applies the 2D weighting factor (n + 1).
        
        Returns:
            recapture_probs_2D: 2D weighted thermal average
        """
        
        # 1. Boltzmann Weights (1D)
        ho_energies = (np.arange(self.n_ho_basis) + 0.5)*self.omega
        beta = 1.0/temp_natural if temp_natural > 0 else np.inf
        
        if np.isinf(beta):
            # If T=0, 2D weighting is trivial (only ground state exists, n=0, factor=1)
            weights_2d = np.zeros(self.n_ho_basis)
            weights_2d[0] = 1.0
        else:
            # If T>0 we use the Boltzmann distribution, P_n_1D = exp(-beta*E_n)
            factors = np.exp(-beta*(ho_energies - ho_energies[0]))
            
            # 2. 2D Weights
            # The density of states in 2D is g_n = (n + 1)
            # Weight_2D_unnorm = (n + 1)*exp(-beta*E_n)
            degeneracy_factor = np.arange(self.n_ho_basis) + 1
            factors_2d = factors*degeneracy_factor
            
            # Normalize
            weights_2d = factors_2d/np.sum(factors_2d)
        
        # Identify significant states (using the 2D distribution as it's wider)
        significant_indices = np.where(weights_2d > 1e-7)[0]
        gaussian_bound_wf = self.get_gaussian_bound_states_spatial()
        recapture_probs_2d = []
        
        print(f"Running simulation scan (2D-weighted)...")
        print(f"Tracking {len(significant_indices)} states.")
        
        for i, tau in enumerate(tau_values):
            prob_recap_2d_t = 0.0
            for n_idx in significant_indices:
                # Get weights
                w2 = weights_2d[n_idx]
                
                # --- Physics Simulation 
                psi_init = self.ho_eigenvectors[n_idx]
                psi_t = self.free_evolution(psi_init, tau, self.x_grid)
                
                # Probability of this specific state 'n' being recaptured
                probs_into_gauss = self.compute_overlap_batch(psi_t, gaussian_bound_wf)
                p_n_recap = np.sum(probs_into_gauss)
                
                # --- Averaging ---
                prob_recap_2d_t += w2*p_n_recap
            recapture_probs_2d.append(prob_recap_2d_t)
        return np.array(recapture_probs_2d)

        

def plot_gaussian_eigenstates(tweezer_obj):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    r = tweezer_obj.x_grid
    scale_factor_um = tweezer_waist_m/(tweezer_obj.waist)/um
    r_plot = r*scale_factor_um

    V_gauss = tweezer_obj._gaussian_potential(r)
    V_ho = 0.5*tweezer_obj.omega**2*r**2 - tweezer_obj.U0 
    
    ax.plot(r_plot, V_gauss, 'k-', linewidth=2, label='Gaussian Potential')
    ax.plot(r_plot, V_ho, 'k--', alpha=0.5, label='Harmonic Approx.')
    
    energies = tweezer_obj.gauss_energies
    wavefunctions = tweezer_obj.get_gaussian_bound_states_spatial()
    
    for i in range(0, len(energies), 10): # plot every 10th state
        psi = wavefunctions[i]
        E = energies[i]
        psi_scaled = psi + E 
        
        ax.plot(r_plot, psi_scaled, lw=1.5)
        ax.text(r_plot[len(r_plot)//2 + 20], E, f'n={i}', fontsize=9, verticalalignment='bottom')

    ax.set_title("Gaussian Trap Eigenstates")
    ax.set_xlabel(r'Position [$\mu$m]')
    ax.set_ylabel(r'Energy [$\hbar\omega$]')
    ax.set_ylim(-tweezer_obj.U0*1.1, 5) 
    ax.set_xlim(-tweezer_waist_m/um*2, tweezer_waist_m/um*2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def load_exp_data(idx: str):
    folder_location = r'raw_data/release_recapture/'
    x_data = np.load(folder_location + idx  + 'x.npy')
    y_data = np.load(folder_location + idx + 'av.npy')
    y_errors = np.load(folder_location + idx + 'er.npy')

    # rescale
    y_data = y_data/np.max(y_data)
    y_errors = y_errors/np.max(y_data)
    return x_data, y_data, y_errors


omega = 1.0 
Tweezer = OpticalTweezer(U0, omega, x_grid, n_ho_basis)

# 1. Visualize the States
plot_gaussian_eigenstates(Tweezer)

# 2. Run Scan (Compute both 1D and 2D)
tau_values = np.linspace(0, max_release_time_s*trap_freq_rad, nr_tau_values)
recap_2d = Tweezer.thermal_recapture_scan_2D_weighted(tau_values, t_natural)

# 3. Plot Recapture Comparison
fig, ax = plt.subplots(figsize=(8,5))
tau_us = tau_values/trap_freq_rad/us

ax.plot(tau_us, recap_2d, 'r.-', linewidth=2, label=f'2D Weighted Model (T = {temperature/uK:.1f} uK)')
if use_exp_data:
    for idx in ['6', '7', '8', '9']:
        x, y, e = load_exp_data(idx)
        ax.errorbar(x, y, yerr=e, fmt='o')

ax.set_xlabel(r'Release Time ($\mu$s)')
ax.set_ylabel('Recapture Probability')
ax.set_ylim(0, 1.05)
ax.set_xlim(0, max_release_time_s/us)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
