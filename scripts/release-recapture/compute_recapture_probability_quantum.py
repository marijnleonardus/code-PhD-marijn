# %% 

"""
Marijn Venderbosch
November 2025

the script uses natural units (hbar = m = 1) for the quantum mechanics calculations, where
omega sets the energy and length scales.

For plotting and physical interpretation, we convert back to SI units using the trap frequency
and atomic mass of strontium-88."""

# import numpy as np
from scipy.constants import pi, Boltzmann, hbar, atomic_mass
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.linalg import eigh
from scipy.integrate import simpson

# append path with 'modules' dir in parent folder
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
utils_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))
sys.path.append(modules_dir)
sys.path.append(utils_dir)

# user defined libraries
from units import kHz, um, uK, us
from plot_utils import Plotting

# %% 

class OpticalTweezer:
    """
    Compute recapture probability for an atom in a Gaussian optical tweezer
    after free evolution.
    """
    
    def __init__(self, U0, omega, r_grid, n_ho_basis=100):
        """
        Parameters:
        -----------
        U0 : float
            Trap depth in units of hbar*omega
        omega : float
            Trap frequency (sets energy scale, typically = 1 in natural units)
        n_ho_basis : int
            Number of harmonic oscillator basis states to use
        """
        self.U0 = U0
        self.omega = omega
        self.a_ho = 1.0/np.sqrt(omega)  # Harmonic oscillator length
        self.n_ho_basis = n_ho_basis
        self.waist = 2*self.a_ho*np.sqrt(U0/omega)  # Beam waist in units of a_ho
        
        # Solve for eigenstates of Gaussian potential
        self.energies, self.eigenstates_ho = self._solve_gaussian_potential(r_grid)
        
    def _harmonic_oscillator_wf(self, n, r):
        """
        Harmonic oscillator wavefunction in position space.
        
        Parameters:
        -----------
        n : int
            Quantum number
        x : array
            Position grid
            
        Returns:
        --------
        psi : array
            Wavefunction values
        """
        norm = 1.0/np.sqrt(2**n*math.factorial(n)*self.a_ho*np.sqrt(pi))
        ri = r/self.a_ho
        H_n = hermite(n)
        return norm*H_n(ri)*np.exp(-ri**2/2)
    
    def _gaussian_potential(self, r):
        """
        Gaussian potential function.
        """
        gaussian_potential = -self.U0*np.exp(-2*r**2/self.waist**2)
        return gaussian_potential
    
    def _solve_gaussian_potential(self, r_grid):
        """
        Solve Schrödinger equation for Gaussian potential in HO basis.
        
        Returns:
        --------
        energies : array
            Eigenvalues (energies in units of hbar*omega)
        eigenstates_ho : array
            Eigenvectors (coefficients in HO basis)
        """
        print(f"Solving Schrödinger equation in {self.n_ho_basis}-state HO basis...")
        
        # Build Hamiltonian matrix in HO basis
        # H = T + gaussian_potential where T is diagonal and gaussian_potential needs integration
        
        # Kinetic + HO potential part (diagonal)
        H = np.diag([(n + 0.5)*self.omega for n in range(self.n_ho_basis)])
        
        # Compute potential matrix elements
        # V_nm = <n|gaussian_potential(x)|m> - <n|0.5*m*omega^2*x^2|m>
        # The second term removes the HO potential that's already in T
        
        dr = r_grid[1] - r_grid[0]
        
        for n in range(self.n_ho_basis):
            psi_n = self._harmonic_oscillator_wf(n, r_grid)
            for m in range(n, self.n_ho_basis):
                psi_m = self._harmonic_oscillator_wf(m, r_grid)
                
                # Full potential
                V_nm = simpson(psi_n*self._gaussian_potential(r_grid)*psi_m, x=r_grid)
                
                # Subtract HO potential that's already included
                V_ho_nm = simpson(psi_n*(0.5*self.omega**2*r_grid**2)*psi_m, x=r_grid)
                
                H[n, m] = H[n, m] + V_nm - V_ho_nm
                if n != m:
                    H[m, n] = H[n, m]
        
        # Diagonalize
        energies, eigenvectors = eigh(H)
        
        # Find bound states (E < 0 relative to trap bottom)
        bound_mask = energies < 0
        n_bound = np.sum(bound_mask)
        
        print(f"Found {n_bound} bound states")
        return energies, eigenvectors
    
    def get_bound_states(self):
        """Return only bound states (E < 0)"""
        bound_mask = self.energies < 0
        return self.energies[bound_mask], self.eigenstates_ho[:, bound_mask]
    
    def position_wavefunction(self, state_idx, r):
        """
        Compute position-space wavefunction for eigenstate.
        
        Parameters:
        -----------
        state_idx : int
            Index of eigenstate
        r : array
            Position grid
            
        Returns:
        --------
        psi : array
            Wavefunction in position space
        """
        psi = np.zeros_like(r)
        coeffs = self.eigenstates_ho[:, state_idx]
        
        for n, c_n in enumerate(coeffs):
            psi += c_n*self._harmonic_oscillator_wf(n, r)
        return psi
    
    def free_evolution(self, tau, x, initial_state_idx=0):
        """
        Compute time-evolved wavefunction after free evolution from a given initial state.
        
        Parameters:
        -----------
        tau : float
            Evolution time in units of 1/omega
        x : array
            Position grid
        initial_state_idx : int
            Index of the initial bound state (default: 0 for ground state)
            
        Returns:
        --------
        psi : array (complex)
            Time-evolved wavefunction
        """
        # Start with specified initial state of the Gaussian potential
        psi_0 = self.position_wavefunction(initial_state_idx, x)
        
        # For general initial state, need to evolve via momentum space
        # or use propagator. For now, let's use the simpler approach:
        # Expand initial state in momentum eigenstates and evolve
        
        dx = x[1] - x[0]
        
        # FFT to momentum space
        psi_k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi_0)))*dx/np.sqrt(2*pi)
        
        # Get momentum grid
        N = len(x)
        dk = 2*pi/(N*dx)
        k = np.fft.fftshift(np.fft.fftfreq(N, dx/(2*pi)))
        
        # Free evolution in momentum space: exp(-i*k^2*tau/2)
        # (using natural units where hbar=m=1)
        phase = np.exp(-1j*k**2*tau/2)
        psi_k_evolved = psi_k*phase
        
        # Transform back to position space
        psi_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psi_k_evolved)))*np.sqrt(2*pi)/dx
        
        # Verify normalization
        norm = simpson(np.abs(psi_t)**2, x=x)
        if abs(norm - 1.0) > 0.01:
            print(f"Warning: evolved wavefunction normalization = {norm:.6f}")
        return psi_t
    
    def compute_overlap(self, psi_evolved, state_idx, r):
        """
        Compute overlap between evolved wavefunction and eigenstate.
        
        Parameters:
        -----------
        psi_evolved : array (complex)
            Time-evolved wavefunction
        state_idx : int
            Index of eigenstate
        x : array
            Position grid
            
        Returns:
        --------
        overlap : complex
            <eigenstate | psi_evolved>
        """
        psi_n = self.position_wavefunction(state_idx, r)
        overlap = simpson(np.conj(psi_n)*psi_evolved, x=r)
        return overlap
    
    def recapture_probability(self, tau, r_grid=None, initial_state_idx=0):
        """
        Compute recapture probability after hold time tau.
        
        Parameters:
        -----------
        tau : float
            Hold time in units of 1/omega
        r_grid : array, optional
            Position grid for integration
        initial_state_idx : int
            Index of the initial bound state (default: 0 for ground state)
            
        Returns:
        --------
        P_n : array
            Probability for each bound state
        P_total : float
            Total recapture probability
        """
        if r_grid is None:
            # Larger grid to capture spreading wavefunction
            # Rule of thumb: wavepacket spreads as sqrt(1 + (omega*tau)^2)
            spread_factor = np.sqrt(1 + (self.omega * tau)**2)
            x_max = max(20*self.a_ho, 15*self.a_ho*spread_factor)
            r_grid = np.linspace(-x_max, x_max, 4000)
        
        # Compute evolved wavefunction from specified initial state
        psi_t = self.free_evolution(tau, r_grid, initial_state_idx)
        
        # Get bound states
        bound_energies, bound_eigenstates = self.get_bound_states()
        n_bound = len(bound_energies)
        
        # Compute overlaps
        probability_n = np.zeros(n_bound)
        for i in range(n_bound):
            overlap = self.compute_overlap(psi_t, i, r_grid)
            probability_n[i] = np.abs(overlap)**2
        probability_total = np.sum(probability_n)
        return probability_n, probability_total
    
    def thermal_recapture_probability(self, tau, temperature_natural, r_grid=None):
        """
        Compute thermally-averaged recapture probability.
        
        The thermal state is a statistical mixture of energy eigenstates with
        Boltzmann weights:
            ρ_thermal = Σ_n p_n |n⟩⟨n|
        where p_n = exp(-E_n/kT) / Z
        
        Parameters:
        -----------
        tau : float
            Hold time in units of 1/omega
        temperature_natural : float
            Temperature in units of hbar*omega/k_B
        r_grid : array, optional
            Position grid for integration
            
        Returns:
        --------
        probability_n_thermal : array
            Thermally-averaged probability for each bound state
        total_prob_thermal : float
            Thermally-averaged total recapture probability
        thermal_occupation : array
            Initial thermal occupation probabilities p_n
        """
        # Get bound states
        bound_energies, _ = self.get_bound_states()
        n_bound = len(bound_energies)
        
        # Compute Boltzmann weights for initial thermal distribution
        # p_n = exp(-E_n / kT) / Z
        beta = 1.0/temperature_natural
        boltzmann_factors = np.exp(-beta*bound_energies)
        partition_function = np.sum(boltzmann_factors)
        thermal_occupation = boltzmann_factors / partition_function
        
        print(f"Ground state occupation: {thermal_occupation[0]:.4f}")
        
        # Initialize arrays for thermally-averaged probabilities
        probability_n_thermal = np.zeros(n_bound)
        total_prob_thermal = 0.0
        
        # For each initial state, compute recapture and weight by thermal occupation
        print(f"Computing thermal average over {n_bound} initial states...")
        for n_init in range(n_bound):
            # Only include states with significant occupation (save computation)
            if thermal_occupation[n_init] < 1e-6:
                continue
                
            # Compute recapture probability starting from state n_init
            P_n, P_total = self.recapture_probability(tau, r_grid, initial_state_idx=n_init)
            
            # Add weighted contribution to thermal average
            probability_n_thermal += thermal_occupation[n_init]*P_n
            total_prob_thermal += thermal_occupation[n_init]*P_total
        
        return probability_n_thermal, total_prob_thermal, thermal_occupation
    

# %% compute result

def main():
    """
    Main function to compute and plot recapture probability.
    """

    # Parameters
    trap_frequency = 10*kHz
    trap_depth = 5*uK
    temperature = 2.2*uK  # Temperature for thermal state (set to 0 for pure ground state)
    m = 88*atomic_mass
    n_ho_basis = 50  # Number of HO basis states
    number_r_grid_points = 1024  # Points in position grid
    max_release_time_s = 50*us
    nr_tau_values = 25  # Number of hold time values to compute

    # Derived parameters
    U0 = Boltzmann*trap_depth/(hbar*2*pi*trap_frequency)  # Trap depth in units of hbar*omega
    T_natural = Boltzmann*temperature/(hbar*2*pi*trap_frequency)  # Temperature in natural units
    trap_freq_rad = 2*pi*trap_frequency  # in rad/s
    tweezer_waist_m = 2*np.sqrt(Boltzmann*trap_depth/(m*trap_freq_rad**2))  # in meters
    print(f"Tweezer waist (1/e^2 radius): {tweezer_waist_m/um:.2f} μm")
    print(f"Temperature: {temperature/uK:.2f} μK = {T_natural:.4f} ℏω/k_B")
    r_grid = np.linspace(-20, 20, number_r_grid_points) # Position grid in units of a_ho

    # Create tweezer object and solve for eigenstates
    omega = 1.0  # Trap frequency (natural units)
    Tweezer = OpticalTweezer(U0, omega, r_grid, n_ho_basis)
        
    # Plot potential and some eigenstates
    fig1, ax1 = plt.subplots()
    ho_length_m = np.sqrt(hbar/(m*2*pi*trap_frequency))  # in meters
    r_grid_plot = r_grid*ho_length_m  # in units of m for plotting
    
    # Plot gaussian potential and HO approximation
    gaussian_potential = Tweezer._gaussian_potential(r_grid)
    ax1.plot(r_grid_plot/um, gaussian_potential, 'k-', linewidth=2, label='Gaussian potential')
    V_ho = 0.5*omega**2*r_grid**2 - U0
    ax1.plot(r_grid_plot/um, V_ho, 'b--', label='Harmonic approximation')
    
    # Plot energy levels
    bound_energies, _ = Tweezer.get_bound_states()

    # Plot n=5, n=10, n=15 wavefunctions, etc. 
    for i in range(0, len(bound_energies), 5):
        psi = Tweezer.position_wavefunction(i, r_grid)
        ax1.plot(r_grid_plot/um, psi + bound_energies[i], label=f'n={i}')
    
    ax1.set_xlabel(r'Radial position $r$ [$\mu$m]')
    ax1.set_ylabel('Energy [$\hbar\omega$]')
    ax1.set_ylim([-U0 - 3, 3])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    Plot = Plotting('output')
    Plot.savefig('release_recapture/tweezer_potential_eigenstates.pdf')
    
    # Plot 2: Recapture probability vs hold time
    fig2, ax2 = plt.subplots()

    tau_values = np.linspace(0, max_release_time_s*trap_freq_rad, nr_tau_values)
    P_total_values_T0 = []
    P_total_values_thermal = []
    
    print("\nComputing recapture vs hold time...")
    
    # Determine if we should compute thermal or T=0 case
    use_thermal = (temperature > 0) and (T_natural > 0.01)

    if use_thermal:
        bound_energies, _ = Tweezer.get_bound_states()
        n_bound = len(bound_energies)
        beta = 1.0/T_natural
        boltzmann_factors = np.exp(-beta*bound_energies)
        partition_function = np.sum(boltzmann_factors)
        thermal_occ = boltzmann_factors/partition_function
    
    for tau in tau_values:
        if use_thermal:
            # Thermal state calculation - compute weighted average
            P_n_thermal = np.zeros(n_bound)
            P_total_thermal = 0.0
            
            for n_init in range(n_bound):
                if thermal_occ[n_init] < 1e-6:
                    continue
                P_n, P_total = Tweezer.recapture_probability(tau, r_grid, initial_state_idx=n_init)
                P_n_thermal += thermal_occ[n_init]*P_n
                P_total_thermal += thermal_occ[n_init]*P_total
            
            P_total_values_thermal.append(P_total_thermal)
        else:
            # Ground state only (T=0)
            _, P_total = Tweezer.recapture_probability(tau, r_grid, initial_state_idx=0)
            P_total_values_T0.append(P_total)

    # Convert dimensionless time to seconds
    tau_seconds = tau_values/trap_freq_rad
    
    if use_thermal:
        ax2.plot(tau_seconds/us, P_total_values_thermal, 'r-', label=rf'T = {temperature/uK:.1f} $\mu$K')
    else:
        ax2.plot(tau_seconds/us, P_total_values_T0, 'b-', label='T = 0 (ground state)')
    
    ax2.set_xlabel(r'Hold time [$\mu$s]')
    ax2.set_ylabel('Recapture Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    Plot.savefig('release_recapture/recapture_vs_time.pdf')
    
    # Plot 3: State-by-state probabilities for specific tau
    fig3, ax3 = plt.subplots()
    tau_example_s = 10*us 
    tau_example = tau_example_s*trap_freq_rad  # in dimensionless units
    
    if use_thermal:
        p_n, p_total, thermal_occ = Tweezer.thermal_recapture_probability(tau_example, T_natural, r_grid)
        title_str = (
            f"T = {temperature/uK:.1f} $\\mu$K, "
            f"$\\tau$ = {tau_example_s/us:.1f} $\\mu$s\n"
            f"Recapture: {p_total:.3f}"
        )
    else:
        p_n, p_total = Tweezer.recapture_probability(tau_example, r_grid, initial_state_idx=0)
        title_str = rf'T = 0, $\tau$ = {tau_example_s/us:.1f} $\mu$s \n Recapture: {p_total:.3f}'
    
    n_states = len(p_n) 
    ax3.bar(range(n_states), p_n[:n_states])
    ax3.set_xlabel(r'Oscillator level $n$')
    ax3.set_ylabel('Probability')
    ax3.set_title(title_str)
    ax3.grid(True, alpha=0.3, axis='y')
    Plot.savefig('release_recapture/state_distribution.pdf')
    
    # Plot 4: Initial thermal occupation (if thermal)
    if use_thermal:
        fig4, ax4 = plt.subplots()
        ax4.bar(range(len(thermal_occ)), thermal_occ, alpha=0.7, color='orange')
        ax4.set_xlabel(r'Oscillator level $n$')
        ax4.set_ylabel('Thermal occupation probability')
        ax4.set_title(rf'Initial thermal distribution at T = {temperature/uK:.1f} $\mu$K')
        ax4.grid(True, alpha=0.3, axis='y')
        Plot.savefig('release_recapture/thermal_occupation.pdf')
    
    plt.show() 


if __name__ == "__main__":
    main()


# %%