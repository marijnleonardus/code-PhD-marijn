
# %% 

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
from single_atoms_class import SingleAtoms
from units import kHz, um, uK
from plot_utils import Plotting

# Physical constants (using natural units where hbar = m = 1)
# In these units, omega defines the energy and length scales

# %% 

class OpticalTweezer:
    """
    Compute recapture probability for an atom in a Gaussian optical tweezer
    after free evolution.
    """
    
    def __init__(self, U0, omega, n_ho_basis=100):
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
        self.energies, self.eigenstates_ho = self._solve_gaussian_potential()
        
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
    
    def _solve_gaussian_potential(self):
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
        
        r_grid = np.linspace(-10*self.a_ho, 10*self.a_ho, 1000)
        dx = r_grid[1] - r_grid[0]
        
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
        print(f"Ground state energy: {energies[0]:.4f} ℏω")
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
    
    def free_evolution(self, tau, x):
        """
        Compute time-evolved wavefunction after free evolution from ground state.
        
        Parameters:
        -----------
        tau : float
            Evolution time in units of 1/omega
        x : array
            Position grid
            
        Returns:
        --------
        psi : array (complex)
            Time-evolved wavefunction
        """
        # Start with ground state of the Gaussian potential (not HO!)
        psi_0 = self.position_wavefunction(0, x)
        
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
            print(f"Warning: evolved wavefunction norm = {norm:.6f}")
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
        overlap = simpson(np.conj(psi_n)*psi_evolved, r=r)
        return overlap
    
    def recapture_probability(self, tau, r_grid=None):
        """
        Compute recapture probability after hold time tau.
        
        Parameters:
        -----------
        tau : float
            Hold time in units of 1/omega
        r_grid : array, optional
            Position grid for integration
            
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
        
        # Compute evolved wavefunction
        psi_t = self.free_evolution(tau, r_grid)
        
        # Get bound states
        bound_energies, bound_eigenstates = self.get_bound_states()
        n_bound = len(bound_energies)
        
        # Compute overlaps
        P_n = np.zeros(n_bound)
        for i in range(n_bound):
            overlap = self.compute_overlap(psi_t, i, r_grid)
            P_n[i] = np.abs(overlap)**2
        P_total = np.sum(P_n)
        return P_n, P_total
    

# %% compute result

def main():
    """
    Main function to compute and plot recapture probability.
    """

    # Parameters
    trap_frequency = 25*kHz
    trap_depth = 50*uK
    m = 88*atomic_mass
    n_ho_basis = 50  # Number of HO basis states
    number_x_grid_points = 2048  # Points in position grid

    # Derived parameters
    U0 = Boltzmann*trap_depth/(hbar*2*pi*trap_frequency)  # Trap depth in units of hbar*omega
    trap_freq_rad = 2*pi*trap_frequency  # in rad/s
    tweezer_waist_m = 2*np.sqrt(Boltzmann*trap_depth/(m*trap_freq_rad**2))  # in meters
    print(f"Tweezer waist (1/e^2 radius): {tweezer_waist_m/um:.2f} μm")

    # Create tweezer object and solve for eigenstates
    omega = 1.0  # Trap frequency (natural units)
    tweezer = OpticalTweezer(U0, omega, n_ho_basis)
    
    # Position grid for plotting and integration
    r_grid = np.linspace(-20, 20, number_x_grid_points)
    
    # Plot potential and some eigenstates
    fig1, ax1 = plt.subplots()
    ho_length_m = np.sqrt(hbar/(m*2*pi*trap_frequency))  # in meters
    r_grid_plot = r_grid*ho_length_m  # in units of m for plotting
    
    # Plot gaussian potential and HO approximation
    gaussian_potential = tweezer._gaussian_potential(r_grid)
    ax1.plot(r_grid_plot/um, gaussian_potential, 'k-', linewidth=2, label='Gaussian potential')
    V_ho = 0.5*omega**2*r_grid**2 - U0
    ax1.plot(r_grid_plot/um, V_ho, 'r--', alpha=0.5, label='Harmonic approximation')
    
    # Plot energy levels
    bound_energies, _ = tweezer.get_bound_states()

    # Plot n=5, n=10, n=15 wavefunctions, etc. 
    for i in range(0, len(bound_energies), 5):
        psi = tweezer.position_wavefunction(i, r_grid)
        ax1.plot(r_grid_plot/um, psi + bound_energies[i], label=f'n={i}')
    
    ax1.plot(r_grid_plot/um, gaussian_potential, 'k-', alpha=0.3, linewidth=1)
    ax1.set_xlabel(r'Radial position $r$ [μm]')
    ax1.set_ylabel('Energy (units of $\hbar\omega$)')
    ax1.set_ylim([-U0 - 3, 3])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    """ # Plot 3: Recapture probability vs hold time
    ax = axes[1, 0]
    tau_values = np.linspace(0, 4*pi, 100)
    P_total_values = []
    
    print("\nComputing recapture vs hold time...")
    for tau in tau_values:
        _, P_total = tweezer.recapture_probability(tau)
        P_total_values.append(P_total)

    # Convert dimensionless time to seconds
    tau_seconds = tau_values / omega_rad_per_sec
    
    ax.plot(tau_seconds/1e-6, P_total_values, 'b-', linewidth=2)
    ax.set_xlabel('Hold time (μs)', fontsize=12)
    ax.set_ylabel('Recapture Probability', fontsize=12)
    ax.set_title('Recapture Probability vs Hold Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 4: State-by-state probabilities for specific tau
    ax = axes[1, 1]
    tau_example = 0.5
    P_n, P_total = tweezer.recapture_probability(tau_example, r_grid)
    
    #n_states = min(30, len(P_n))
    n_states = len(P_n)
    ax.bar(range(n_states), P_n[:n_states], color='steelblue', alpha=0.7)
    ax.set_xlabel('Quantum Number n', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f'State Distribution at $\\tau\omega = {tau_example}$\n' + f'Total Recapture: {P_total:.4f}')
    ax.grid(True, alpha=0.3, axis='y')
    plt.show() 
    
    # Print detailed results for specific hold time
    print(f"\n{'='*60}")
    print(f"Detailed Results for τω = {tau_example}")
    print(f"{'='*60}")
    print(f"Total recapture probability: {P_total:.6f}")"""


if __name__ == "__main__":
    main()

# %%
