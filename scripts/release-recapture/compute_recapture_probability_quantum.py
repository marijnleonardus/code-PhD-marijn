
"""
Marijn Venderbosch 
November 2025

Code for computing recapture probability after turning off tweezers for variable duration

Assuming Gaussian optical tweezer potential. 

Inputs to the model are Trap depth and trap frequency (or waist).

You can also provide a temperature. If not provided, will assume n = 0 state initially.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.constants import pi, Boltzmann, hbar, atomic_mass
import math
import time
from scipy.interpolate import interp1d

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
trap_depth = 190*uK
temperature = 18*uK
waist_computed = 0.92*um
m = 85*atomic_mass
n_ho_basis = 110  # should be higher than number of bound states
number_r_grid_points = int(2**12) # Use power of 2 for FFT efficiency
max_radius = 5  # in harmonic oscillator units
max_release_time_s = 100*us
nr_tau_values = 30 
use_exp_data = True  # Whether to load and plot experimental data

# --- Derived parameters ---
trap_frequency = 2*np.sqrt(Boltzmann*trap_depth/(m*waist_computed**2))/(2*pi)  # in Hz
print(trap_frequency)
U0 = Boltzmann*trap_depth/(hbar*2*pi*trap_frequency)
t_natural = Boltzmann*temperature/(hbar*2*pi*trap_frequency)
trap_freq_rad = 2*pi*trap_frequency

# Calculate waist for reporting
tweezer_waist_m = 2*np.sqrt(Boltzmann*trap_depth/(m*trap_freq_rad**2))
print(f"Tweezer waist (1/e^2 radius): {tweezer_waist_m/um:.2f} μm")

# Grid setup
# We use a slightly larger grid to ensure we capture the tails
spread_factor = np.sqrt(1 + (1.0*max_release_time_s*trap_freq_rad)**2)
x_max = max(max_radius, max_radius*spread_factor) 
r_grid = np.linspace(-x_max, x_max, number_r_grid_points)


class OpticalTweezer:
    """
    Compute recapture probability for an atom in a Gaussian optical tweezer
    after free evolution. Optimized using vectorization.
    """
    
    def __init__(self, U0, omega, r_grid, n_ho_basis=100):
        self.U0 = U0
        self.omega = omega
        self.a_ho = 1.0/np.sqrt(omega)
        self.n_ho_basis = n_ho_basis
        self.waist = 2*self.a_ho*np.sqrt(U0/omega)
        
        # Pre-compute normalization constants for Hermite polynomials
        # to avoid re-calculating factorials constantly
        self._norms = np.array([1.0/np.sqrt(2**n*math.factorial(n)*self.a_ho*np.sqrt(pi)) 
            for n in range(n_ho_basis)])
        
        self.energies, self.eigenstates_ho = self._solve_gaussian_potential(r_grid)
        
    def _harmonic_oscillator_wf_batch(self, r):
        """
        Stable computation of HO wavefunctions ψ_n(x)
        using direct recurrence on ψ_n (not H_n).
        """
        x = r/self.a_ho
        N0 = self._norms[0]
        N1 = self._norms[1]

        basis_states = np.zeros((self.n_ho_basis, len(r)))

        # psi_0
        psi_prev = N0*np.exp(-x**2/2)
        basis_states[0] = psi_prev

        if self.n_ho_basis > 1:
            # psi_1
            psi_curr = N1*(2*x)*np.exp(-x**2/2)
            basis_states[1] = psi_curr

            for n in range(1, self.n_ho_basis - 1):
                # stable recurrence for ψ_n
                psi_next = (np.sqrt(2/(n+1))*x*psi_curr - np.sqrt(n/(n+1))*psi_prev)
                basis_states[n+1] = psi_next
                psi_prev, psi_curr = psi_curr, psi_next
        return basis_states
    
    def _gaussian_potential(self, r):
        return -self.U0*np.exp(-2*r**2/self.waist**2)
    
    def _solve_gaussian_potential(self, r_grid):
        print(f"Solving Schrödinger equation in {self.n_ho_basis}-state HO basis (Vectorized)...")
        t_start = time.time()
        
        # 1. Kinetic + HO potential (Diagonal)
        H = np.diag([(n + 0.5)*self.omega for n in range(self.n_ho_basis)])
        
        # 2. Compute Potential Matrix Elements via Matrix Multiplication
        # We need <n| V_gauss - V_ho |m>
        
        # Precompute all basis functions: Shape (N_basis, N_grid)
        psi_matrix = self._harmonic_oscillator_wf_batch(r_grid)
        
        # The perturbation potential V(r) - V_ho(r)
        V_perturbation = self._gaussian_potential(r_grid) - (0.5*self.omega**2*r_grid**2)
        
        # Integration weight
        dr = r_grid[1] - r_grid[0]
        
        # Vectorized Integration:
        # V_nm = sum_x (psi_n(x)*V_diff(x)*psi_m(x))*dx
        # This is equivalent to: Psi @ diag(V) @ Psi.T
        
        # First, multiply rows of Psi by V_perturbation (broadcasting)
        weighted_psi = psi_matrix*V_perturbation[np.newaxis, :]
        
        # Then matrix multiply: (N, Grid) @ (Grid, N) -> (N, N)
        V_matrix = np.dot(weighted_psi, psi_matrix.T)*dr
        
        H += V_matrix
        
        # Diagonalize
        energies, eigenvectors = eigh(H)
        
        t_end = time.time()
        bound_mask = energies < 0
        n_bound = np.sum(bound_mask)
        print(f"Found {n_bound} bound states. Matrix construction & diag took {t_end-t_start:.4f}s")
        
        return energies, eigenvectors
    
    def get_bound_states(self):
        bound_mask = self.energies < 0
        return self.energies[bound_mask], self.eigenstates_ho[:, bound_mask]
    
    def position_wavefunction(self, state_idx, r):
        # Optimized linear combination
        coeffs = self.eigenstates_ho[:, state_idx]
        psi_matrix = self._harmonic_oscillator_wf_batch(r)
        # sum_n (c_n*psi_n(r)) -> dot product
        return np.dot(coeffs, psi_matrix)

    def free_evolution(self, tau, x, initial_state_idx=0):
        psi_0 = self.position_wavefunction(initial_state_idx, x)
        dx = x[1] - x[0]
        
        # FFT Evolution
        psi_k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_0)))*dx/np.sqrt(2*pi)
        
        N = len(x)
        k = np.fft.fftshift(np.fft.fftfreq(N, dx/(2*pi)))
        
        phase = np.exp(-1j*k**2*tau/2)
        psi_k_evolved = psi_k*phase
        
        psi_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psi_k_evolved)))*np.sqrt(2*pi)/dx
        return psi_t
    
    def recapture_probability(self, tau, r_grid, initial_state_idx=0):
        """compute recapture probability into bound states after free evolution,
        for a given initial state"""

        # We assume r_grid is passed correctly (not None) for speed
        psi_t = self.free_evolution(tau, r_grid, initial_state_idx)
        
        # Get bound states
        bound_energies, bound_eigenstates = self.get_bound_states()
        
        # Project evolved state onto ALL bound states at once
        # <n|psi_t> = integral(psi_n(x)* psi_t(x))
        # First get all bound state spatial wavefunctions
        
        # Note: calling position_wavefunction loop is slow. 
        # Better: project psi_t onto HO basis, then multiply by eigenvectors.
        # But for now, let's just optimize the overlap integration.
        
        dx = r_grid[1] - r_grid[0]
        
        # Calculate overlap with every bound state
        # We can reconstruct the bound states matrix (Grid, n_bound)
        # but simpler is just doing the dot product with the precomputed HO basis
        
        # 1. Project psi_t onto HO basis: c_n(t) = <n_HO | psi_t>
        ho_basis = self._harmonic_oscillator_wf_batch(r_grid) # (N_HO, Grid)
        c_t_ho = np.dot(ho_basis, psi_t)*dx # (N_HO,)
        
        # 2. Convert to Energy basis: c_E(t) = Eigenvectors.T @ c_t_ho
        # eigenvectors is (N_HO, N_HO), we only want bound ones
        coeffs_bound = bound_eigenstates # (N_HO, n_bound)
        
        # Overlaps in energy basis
        overlaps = np.dot(coeffs_bound.T, np.conj(c_t_ho)) # (n_bound,)
        
        probs = np.abs(overlaps)**2
        return probs, np.sum(probs)

    def thermal_recapture_probability(self, tau, temperature_natural, r_grid):
        """compute recapture probability averaged over thermal distribution"""

        bound_energies, _ = self.get_bound_states()
        n_bound = len(bound_energies)
        
        # calculate Boltzmann factors
        beta = 1.0/temperature_natural
        boltzmann_factors = np.exp(-beta*bound_energies)
        partition_function = np.sum(boltzmann_factors)
        thermal_occupation = boltzmann_factors/partition_function
        
        probability_n_thermal = np.zeros(n_bound)
        total_prob_thermal = 0.0
        
        # the inner functions within this loop are vectorized for speed
        for n_init in range(n_bound):
            if thermal_occupation[n_init] < 1e-6: continue
            
            prob_n, prob_total = self.recapture_probability(tau, r_grid, initial_state_idx=n_init)
            probability_n_thermal += thermal_occupation[n_init]*prob_n
            total_prob_thermal += thermal_occupation[n_init]*prob_total
            
        return probability_n_thermal, total_prob_thermal, thermal_occupation


print("Initializing Tweezer and solving eigenstates...")
omega = 1.0
Tweezer = OpticalTweezer(U0, omega, r_grid, n_ho_basis)


def plot_potential_and_states():
    fig, ax = plt.subplots()
    
    # Conversion for x-axis
    ho_length_m = np.sqrt(hbar/(m*2*pi*trap_frequency))
    r_grid_plot = r_grid*ho_length_m 
    
    # 1. Plot gaussian potential
    gaussian_potential = Tweezer._gaussian_potential(r_grid)
    ax.plot(r_grid_plot/um, gaussian_potential, 'k-', linewidth=2, label='Gaussian potential')
    
    # 2. Plot Harmonic approximation
    V_ho = 0.5*omega**2*r_grid**2 - U0
    ax.plot(r_grid_plot/um, V_ho, 'k--', alpha=0.8, label='Harmonic approximation')
    
    # 3. Plot Wavefunctions offset by their Energy
    bound_energies, _ = Tweezer.get_bound_states()
    
    # Loop through states (0, 5, 10...)
    print("Plotting wavefunctions...")
    for i in range(0, len(bound_energies), 10):
        psi = Tweezer.position_wavefunction(i, r_grid)
        
        # Scaling psi slightly for better visibility if needed, 
        # or keeping it raw as in your original script:
        ax.plot(r_grid_plot/um, psi + bound_energies[i], label=f'n={i}')

    ax.set_xlabel(r'Radial position $r$ [$\mu$m]')
    ax.set_ylabel('Energy [$\hbar\omega$]')
    ax.set_ylim([-U0 - 3, 5]) # Adjusted limits to see the potential bottom
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Trap Potential & Eigenstates (Depth = {trap_depth/uK:.1f} $\mu$K)')


def load_exp_data():
    folder_location = r'raw_data/release_recapture/'
    x_data = np.load(folder_location + '8x.npy')
    y_data = np.load(folder_location + '8av.npy')
    y_errors = np.load(folder_location + '8er.npy')

    # rescale
    y_data = y_data/np.max(y_data)
    y_errors = y_errors/np.max(y_data)
    return x_data, y_data, y_errors


def plot_recapture_probability(tau_values, prob_tot_values):
    """Plotting the simulation and experimental data together"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # plot theory curve
    tau_seconds = tau_values/trap_freq_rad
    ax.plot(tau_seconds/us, prob_tot_values, 'r.-', label=rf'T = {temperature/uK:.2f} $\mu$K (Simulation)')

    # plot experimental data
    if use_exp_data:
        x_exp, y_exp, y_err_exp = load_exp_data()
        ax.errorbar(x_exp, y_exp, yerr=y_err_exp, ls='none', label='Exp. data')

        # calculate correlation coefficient
        # Interpolate theory onto the experimental x-values
        internpol_function = interp1d(tau_seconds/us, prob_tot_values, kind='quadratic')
        theory_interp = internpol_function(x_exp)

        # Compute R^2 goodness of fit
        residuals = y_exp - theory_interp
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_exp - np.mean(y_exp))**2)
        r_squared = 1 - ss_res/ss_tot
        print(f"R² goodness-of-fit (theory vs experiment): {r_squared:.4f}")
    
    # formatting
    ax.set_xlabel(r'Hold time [$\mu$s]')
    ax.set_ylabel('Recapture Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])


def main():
    plot_potential_and_states()

    # --- Calculation: Recapture vs Hold time ---
    print(f"\nComputing thermal recapture ({nr_tau_values} time points)...")
    tau_values = np.linspace(0, max_release_time_s*trap_freq_rad, nr_tau_values)
    
    t0 = time.time()
    prob_tot_values = []
    
    for tau in tau_values:
        if temperature > 0*uK:
            _, p_tot, _ = Tweezer.thermal_recapture_probability(tau, t_natural, r_grid)
            prob_tot_values.append(p_tot)
        else:
            _, p_tot = Tweezer.recapture_probability(tau, r_grid)
            prob_tot_values.append(p_tot)
    print(f"Calculation complete in {time.time()-t0:.2f} seconds.")

    plot_recapture_probability(tau_values, prob_tot_values)
    plt.show()


if __name__ == "__main__":
    main()
