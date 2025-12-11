"""
Marijn Venderbosch 
November 2025

Code for computing recapture probability after turning off tweezers for variable duration.

* Initialize atom as thermal distribution harmonic oscillator states
* Free evolve each state for time tau
* Compute recapture probability into bound states of Gaussian potential, found by diagonalization of the Hamiltonian

If you have experimental data, you can load it and compute R^2 values for the fit.

If you want to run this code yourself, it is easiest to clone the full repository, as the code
relies on imports from other files in the repository.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import eigh
from scipy.constants import pi, Boltzmann, hbar, atomic_mass
import math
import os
import sys

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir,'../../raw_data/release_recapture/') 
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
utils_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))

sys.path.append(modules_dir)
sys.path.append(utils_dir)
sys.path.append(data_dir)

from units import uK, um, us, kHz
from plot_utils import Plotting
from statistics_utils import Stats
from atoms_tweezer_class import AtomicMotion

# --- Parameters ---
# physics parameters. Specify either waist, or trap frequency. One of them can be 'None'
trap_depth = 190*uK*.75/.82
tweezer_waist_m = 0.92*um
trap_frequency = None #45*kHz # 82*kHz
m = 85*atomic_mass
temperatures_to_scan = np.array([0, 2.5])*uK

# simulation parameters
n_ho_basis = 150  # number of harmonic oscillator basis states to use
number_x_grid_points = int(2**15) # discretization grid points, use power of 2 for FFT efficiency
max_radius = 9  # in HO units, geometric cutoff for spatial grid
max_release_time_s = 100*us
nr_tau_values = 50  # number of release times to simulate

# loading exp. data
use_exp_data = True 
folder_location = r'raw_data/release_recapture/'
dataset_name = '6'  # which dataset to load

# --- Derived parameters ---
# trap frequency and tweezer waist are related, and follow from each other
# in combination with the specified trap depth in Kelvin
if trap_frequency is None:
    trap_frequency = AtomicMotion.trap_frequency_radial(m, tweezer_waist_m, trap_depth)
    print(f"calc. trap Frequency: {trap_frequency/kHz:.2f} kHz")
if tweezer_waist_m is None:
    tweezer_waist_m = AtomicMotion.waist_from_trap_frequency(m, trap_frequency, trap_depth)
    print(f"calc. tweezer waist: {tweezer_waist_m/um:.2f} um")
trapdepth_dl = Boltzmann*trap_depth/(hbar*2*pi*trap_frequency) 
trap_freq_rad = 2*pi*trap_frequency

# Grid setup, calculate spatial grid in dimensionless units (dl)
spread_factor = np.sqrt(1 + (1.0*max_release_time_s*trap_freq_rad)**2)
x_max_dl = max(max_radius, max_radius*spread_factor) 
x_grid_dl = np.linspace(-x_max_dl, x_max_dl, number_x_grid_points)


class RecaputureOpticalTweezer:
    """
    Compute recapture probability for an atom in a Gaussian optical tweezer.
    """
    
    def __init__(self, trapdepth_dl, omega_dl, x_grid_dl, n_ho_basis=100):
        self.trapdepth_dl = trapdepth_dl
        self.omega_dl = omega_dl
        self.x_grid_dl = x_grid_dl
        self.dx = x_grid_dl[1] - x_grid_dl[0]
        
        self.a_ho = 1.0/np.sqrt(omega_dl)
        self.n_ho_basis = n_ho_basis
        self.waist = 2*self.a_ho*np.sqrt(trapdepth_dl/omega_dl)
        
        # calculate normalization constants for HO wavefunctions
        self._norms = np.array([1.0/np.sqrt(2**n*math.factorial(n)*self.a_ho*np.sqrt(pi)) 
            for n in range(n_ho_basis)])
        
        print("Solving Gaussian eigenstates...")
        self.gauss_energies, self.gauss_eigenvectors = self._solve_gaussian_potential(x_grid_dl)
        
        print("Pre-computing HO basis states...")
        self.ho_eigenvectors = self._harmonic_oscillator_wf_batch(x_grid_dl)

    def _harmonic_oscillator_wf_batch(self, x):
        """computation of HO wavefunctions"""
        x = x/self.a_ho
        basis_states = np.zeros((self.n_ho_basis, len(x)))
        psi_prev = self._norms[0]*np.exp(-x**2/2)
        basis_states[0] = psi_prev

        if self.n_ho_basis > 1:
            # the wavefunctions are built up recursively
            psi_curr = self._norms[1]*(2*x)*np.exp(-x**2/2)
            basis_states[1] = psi_curr
            for n in range(1, self.n_ho_basis - 1):
                psi_next = (np.sqrt(2/(n + 1))*x*psi_curr - np.sqrt(n/(n + 1))*psi_prev)
                basis_states[n + 1] = psi_next
                psi_prev, psi_curr = psi_curr, psi_next
        return basis_states
    
    def _gaussian_potential(self, x):
        """1d gaussian potential"""
        gaussian_1d = -self.trapdepth_dl*np.exp(-2*x**2/self.waist**2)
        return gaussian_1d
    
    def _solve_gaussian_potential(self, x_grid_dl):
        """Diagonalize Hamiltonian in HO basis to find Gaussian bound states."""
        # Build Hamiltonian 
        H = np.diag([(n + 0.5)*self.omega_dl for n in range(self.n_ho_basis)])
        psi_matrix = self._harmonic_oscillator_wf_batch(x_grid_dl)
        V_perturbation = self._gaussian_potential(x_grid_dl) - (0.5*self.omega_dl**2*x_grid_dl**2)
        
        weighted_psi = psi_matrix*V_perturbation[np.newaxis, :]
        V_matrix = np.dot(weighted_psi, psi_matrix.T)*self.dx
        H += V_matrix
        
        # Diagonalize
        energies, eigenvectors = eigh(H)

        # Find bound states
        bound_mask = energies < 0
        return energies[bound_mask], eigenvectors[:, bound_mask]

    def get_gaussian_bound_states_spatial(self):
        """Get Gaussian bound states in HO basis states."""
        coeffs = self.gauss_eigenvectors
        basis = self.ho_eigenvectors
        return np.dot(coeffs.T, basis)

    def free_evolution(self, psi_0, tau, x):
        """compute free evolution of wavefunction psi_0 for time tau using propagator
        The propagator is calculated in momentum space (p^2/2m)
        
        calculation to and from momentum space is done via FFT"""
        dx = x[1] - x[0]

        # convert to momentum space
        psi_k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_0)))*dx/np.sqrt(2*pi)
        
        # evolve wavefunction in momentum space
        N = len(x)
        k = np.fft.fftshift(np.fft.fftfreq(N, dx/(2*pi)))
        phase = np.exp(-1j*k**2*tau/2)
        psi_k_evolved = psi_k*phase

        # convert back to spatial space: \phi(x,t)
        psi_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psi_k_evolved)))*np.sqrt(2*pi)/dx
        return psi_t

    def compute_overlap_batch(self, psi_evolved, bound_states_spatial):
        """Compute overlap of evolved state with multiple bound states at once."""
        overlaps = np.dot(bound_states_spatial, psi_evolved)*self.dx
        overlaps_abs_squared = np.abs(overlaps)**2
        return overlaps_abs_squared

    def thermal_recapture_scan_2D_weighted(self, tau_values, temp_natural):        
        """Computes 1D recapture probability and extend to 2Dusing the density of states
        for a 2D harmonic oscillator (n+1), then renormalizes.
        """
        # Boltzman factors (1d)
        ho_energies = (np.arange(self.n_ho_basis) + 0.5)*self.omega_dl
        beta = 1.0/temp_natural if temp_natural > 0 else np.inf
        
        if np.isinf(beta):
            # If T=0, 2D weighting is trivial (only ground state exists, n=0, factor=1)
            weights_2d = np.zeros(self.n_ho_basis)
            weights_2d[0] = 1.0
        else:
            # If T>0 we use the Boltzmann distribution, P_n_1D = exp(-beta*E_n)
            factors = np.exp(-beta*(ho_energies - ho_energies[0]))

            # The density of states in 2D is g_n = (n + 1)
            # Weight_2D_unnorm = (n + 1)*exp(-beta*E_n)
            degeneracy_factor = np.arange(self.n_ho_basis) + 1
            factors_2d = factors*degeneracy_factor
            weights_2d = factors_2d/np.sum(factors_2d)
        
        # check in which HO states the wavefunction 'lives' and only consider those
        significant_indices = np.where(weights_2d > 1e-7)[0]
        gaussian_bound_wf = self.get_gaussian_bound_states_spatial()
        recapture_probs_2d = []
        
        print(f"Running simulation scan (2D-weighted)...")
        print(f"Tracking {len(significant_indices)} states.")   

        for i, tau in enumerate(tau_values):
            prob_recap_2d_t = 0.0
            for n_idx in significant_indices:
                w2 = weights_2d[n_idx]
                psi_init = self.ho_eigenvectors[n_idx]

                # evolve state
                psi_t = self.free_evolution(psi_init, tau, self.x_grid_dl)

                # Probability of this specific state 'n' being recaptured
                probs_into_gauss = self.compute_overlap_batch(psi_t, gaussian_bound_wf)
                p_n_recap = np.sum(probs_into_gauss)

                # averaging
                prob_recap_2d_t += w2*p_n_recap
            recapture_probs_2d.append(prob_recap_2d_t)
        return np.array(recapture_probs_2d)


def plot_gaussian_eigenstates(tweezer_obj):
    """Plot Gaussian potential and some of its bound states."""
    fig, ax = plt.subplots(figsize=(5, 3))
    
    r = tweezer_obj.x_grid_dl
    scale_factor_um = tweezer_waist_m/(tweezer_obj.waist)/um
    r_plot = r*scale_factor_um

    V_gauss = tweezer_obj._gaussian_potential(r)
    V_ho = 0.5*tweezer_obj.omega_dl**2*r**2 - tweezer_obj.trapdepth_dl 
    
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
    ax.set_xlabel(r'Position [$\mu$m]')
    ax.set_ylabel(r'Energy [$\hbar\omega$]')
    ax.set_ylim(-tweezer_obj.trapdepth_dl*1.1, 5) 
    ax.set_xlim(-tweezer_waist_m/um*2, tweezer_waist_m/um*2)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def load_exp_data(idx: str):
    """load experimental data from specified folder, 
    x,y and y_err are stored in separate .npy files"""
    x_data = np.load(folder_location + idx  + 'x.npy')
    y_data = np.load(folder_location + idx + 'av.npy')
    y_errors = np.load(folder_location + idx + 'er.npy')
    return x_data, y_data, y_errors


def main():
    # trap frequency in dimensionless units (dl)
    omega_dl = 1.0 
    Recapture = RecaputureOpticalTweezer(trapdepth_dl, omega_dl, x_grid_dl, n_ho_basis)
    plot_gaussian_eigenstates(Recapture)

    # load experimental data if desired
    exp_x, exp_y, exp_err = None, None, None
    if use_exp_data:
        print("Loading experimental data...")
        exp_x, exp_y, exp_err = load_exp_data(dataset_name)

    # simulation release times in dimensionless units
    tau_values_sim = np.linspace(0, max_release_time_s*trap_freq_rad, nr_tau_values)
    tau_values_sim_us = tau_values_sim/trap_freq_rad/us

    results = {} # Store results: temp -> (y_sim, r_squared)

    print(f"Starting scan for temperatures: {temperatures_to_scan/uK} uK")

    # matrix to store R^2 values
    r_squared_array = []

    # do simulation for each temperature
    for temp in temperatures_to_scan:
        # convert temperature to dimensionless units
        t_natural = Boltzmann*temp/(hbar*trap_freq_rad)

        # run simulation
        recap_prob = Recapture.thermal_recapture_scan_2D_weighted(tau_values_sim, t_natural)
        
        # calculate R^2 if experimental data is available
        r_squared = None
        if use_exp_data and exp_x is not None:
            # recale curve to match (t=0) exp. data, as a result of SPAM errors and finite survival probability
            scale_factor = np.max(exp_y)/recap_prob[0]
            recap_prob *= scale_factor

            # interpolate simulation result onto experimental time points
            # exp_x is in microseconds, simulation x needs to be converted to microseconds
            sim_interpolated = np.interp(exp_x, tau_values_sim_us, recap_prob)
            
            # calculate R^2
            r_squared = Stats.calculate_r_squared(exp_y, sim_interpolated)
            r_squared_array.append(r_squared)
            print(f"T = {temp/uK:5.1f} uK | R^2 = {r_squared:.4f}")
        else:
            print(f"T = {temp/uK:5.1f} uK | Done (No Fit)")
        
        # (optional) store recapture prob. curves for using in different scripts
        filename = f"output/release_recapture/recap_prob_T_{temp/uK:.1f}uK_scaled.txt"
        np.savetxt(filename, recap_prob)

        # store in single object results for easy retrieval further in this script
        results[temp] = (recap_prob, r_squared)

    # plot theory curves and exp. data
    fig, ax = plt.subplots(figsize=(5, 3))

    # plot exp. data
    if use_exp_data and exp_x is not None:
        ax.errorbar(exp_x, exp_y, yerr=exp_err, fmt='ko', capsize=3, label='Exp Data', zorder=10)

        # plot R^2 values for each simulated temperature
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.scatter(temperatures_to_scan/uK, r_squared_array)
        ax2.set_xlabel(r'Temperature ($\mu$K)')
        ax2.set_ylabel(r'$R^2$ fit')

    # plot Simulations
    colors = cm.viridis(np.linspace(0, 1, len(temperatures_to_scan)))
    best_r2 = -np.inf
    for (temp, (y_sim, r2)), color in zip(results.items(), colors):
        label_str = f'T = {temp/uK:.1f} uK'
        if r2 is not None:
            label_str += f' ($R^2$={r2:.3f})'
            if r2 > best_r2:
                best_r2 = r2
        ax.plot(tau_values_sim_us, y_sim, '-', color=color, linewidth=2, alpha=0.8, label=label_str)
    ax.set_xlabel(r'Release Time ($\mu$s)')
    ax.set_ylabel('Recapture Probability')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_release_time_s/us)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
