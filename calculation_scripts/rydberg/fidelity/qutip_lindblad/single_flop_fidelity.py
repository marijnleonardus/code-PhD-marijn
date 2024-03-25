# author: Marijn Venderbosch
# January 2023

# standard modules
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import proton_mass

# custom module
import sys
import os
custom_module_dir = os.path.abspath("../code-PhD-marijn/modules/")
sys.path.append(custom_module_dir)
from atom_class import AtomicMotion

"""starting in the clock state, compute fidelity after doing a pi pulse: 
clock-rydberg, or equivalently: rydberg-clock

only takes into account 3 decoherence mechanisms
- spontaneous emission from rydberg state
- finite laser linewidth, which shows up as a decoherence (dephasing) matrix*
- doppler broadening from atom moving with finite temperature in tweezer**

*for details see PhD thesis of Jonathan Prithard p. 42

**modeled similarly to the laser linewidth contribution as it is also
an error in detuning, which might be an oversimplification"""

plt.style.use('default')


# %% variables

pi = np.pi

# number of pi pulses to run
flops_to_compute = 1

# time matrix in units of Rabi flops / 2pi
time = np.linspace(0.0, flops_to_compute*2*pi, flops_to_compute*1000)  

# laser linewidth
laser_linewidth_array = 2*pi*np.array([1, 10, 100, 1e3, 1e4, 1e5])  # [Hz]

# laser rabi frequency and detuning
laser_rabi_frequency = 2*pi*5e6  # [Hz]
laser_detuning = 0  # [Hz] 

# rydberg lifetime and corresponding decay rate 
rydberg_lifetime = 86e-6  # [s] for n=61 from Madhav paper
rydberg_decay_rate = 1/rydberg_lifetime  # [Hz]

# doppler broadening from atom motion
mass_sr = 88*proton_mass  # [kg]
wavelength=317e-9  # [m]
trapdepth_kelvin = 100e-6  # [K]
beam_waist = 0.8e-6  # [m]
temperature_atom = 100e-6  # [K]

# trap frequency of atom in tweezer
trap_freq = AtomicMotion.trap_frequency_tweezer_radial(mass_sr, beam_waist, trapdepth_kelvin)

# standard dev. of doppler broadening as seen by atom in tweezer
sigma_detuning, _ = AtomicMotion.doppler_broadening_tweezer(mass_sr, temperature_atom, trap_freq, wavelength)
fwhm_detuning_doppler = 2*np.sqrt(2*np.log(2))*sigma_detuning

# %% functions, calculations

def make_dimensionless(number, rabi):
    """
    Parameters
    ----------
    number : float
        frequency to make dimensionless in [Hz]
    rabi_freq : float
        rabi frequency of rydberg laser in [Hz].

    Returns
    -------
    dimenionless number: float
        dimensionless frequency in terms of rabi frequency.
    """
    return number/rabi


def solve_lindblad(linewidth, rabi, detuning, dopplerbroadening, decayrate):
    """
    Parameters
    ----------
    linewidth : float
        linewidth of rydberg laser in [Hz].
    rabi : float
        rabi frequency of clock-rydberg transition in [Hz].
    detuning : float
        detuning of Rydberg laser in [Hz].
    decayrate : float
        decay rate from rydberg state to 3P0,1,2 triplet in [Hz].

    Returns
    -------
    result : qutip object
        contains result of qutip integration like times, expectation values
    """
    
    # divide by rabi frequency to make everything dimensionless
    linewidth = make_dimensionless(linewidth, rabi)
    detuning = make_dimensionless(detuning, rabi)
    decayrate = make_dimensionless(decayrate, rabi)
    dopplerbroadening = make_dimensionless(dopplerbroadening, rabi) 
    rabi = make_dimensionless(rabi, rabi)

    # define hamiltonian decomposed in pauli matrices
    # units are hbar=1 (atomic units)
    hamiltonian = 0.5*(rabi*qt.sigmax() - detuning*qt.sigmaz())

    # two level system starting in |g>=(0,1)
    n=2
    psi0=qt.basis(n, 1)

    # obtain density matrix
    rho0 = qt.ket2dm(psi0)

    # ladder operator
    a = qt.destroy(n)

    # define lindblad operators and the total lindblad superoperator
    lindblad_spont_em = np.sqrt(decayrate)*a

    # define lindblad operator for laser linewidth,
    # see mathematica np file in same folder
    c_matrix = a.dag()*a
    lindblad_linewidth = np.sqrt(linewidth)*c_matrix
    
    # lindblad loss matrix for atomic motion doppler broadening, similar form 
    # as laser linewidth term, as it is also a detuning variation
    lindblad_doppler = np.sqrt(dopplerbroadening)*c_matrix

    # sum lindblad collapse operator contributions
    lindblad_total = [lindblad_spont_em, lindblad_linewidth, lindblad_doppler]

    # solve master equation given hamiltonian, initial state rho0,
    # and collapse operators c_ops
    result = qt.mesolve(hamiltonian, rho0, time, c_ops=lindblad_total)
    return result


def compute_fidelity(linewidth):
    """
    Parameters
    ----------
    linewidth : float
        laser linewidth in [Hz].

    Returns
    -------
    result : qutip object
        result of qutip simulation.
    population_g : float
        probability to be in clock state
    population_r : float
        probability to be in rydberg state.
    fidelity : float
        accaracy of 2pi rydberg pulse
    """
    
    # perform qutip simulation
    result = solve_lindblad(linewidth, laser_rabi_frequency, laser_detuning, fwhm_detuning_doppler, rydberg_decay_rate)
    
    # only compute rydberg and clock populations (diagonal density matrix elements)
    # but others can be uncommented as well
    population_r = np.real(qt.expect(result.states, qt.projection(2, 0, 0)))
    # population_g = np.real(qt.expect(result.states, qt.projection(2, 1, 1)))
    # coherence = np.real(qt.expect(result.states, qt.projection(2, 0, 1)))
    
    # fidelity is defined as rydberg state population after t=pi/rabi pulse
    # fidelity error is 1-fidelity
    error = population_r[-1]
    
    return result, population_r, error


def loop_over_linewidths(linewidths):
    """
    Parameters
    ----------
    linewidths : np.array
        np.array of laser linewidths to solve the qutip simulation for.

    Returns
    -------
    results : np.array
        each entry contains a qutip object.
    populations : np.array
        each entry contains another np array with the population of the rydberg state
        as a function of time.
    fidelities : np.array
        fidelties of 2pi pulses.

    """
    
    # empty arrays to iterate over
    results = []
    fidelities = []
    populations = []

    for linewidth in linewidths:
        # compute fidelity as a functino of linewidth
        result, population_r, fidelity = compute_fidelity(linewidth)
               
        # store fidelity result as well as result objects
        results.append(result)
        fidelities.append(fidelity)
        populations.append(population_r)
        
    return results, populations, fidelities

results, populations, fidelity_errors = loop_over_linewidths(laser_linewidth_array)

    
# %% Plotting

fig, ax = plt.subplots(figsize=(4, 3))
ax.grid()

# only plot rydberg population, but others can be uncommented as well
ax.plot(results[0].times/(2*pi), populations[0], label=r'$\rho_{rr}$')
# ax.plot(result.times, population_g, label=r'$\rho_{gg}$')
# ax.plot(result.times, coherence, label=r'$\rho_{eg}$')

ax.set_title(r'Population of $|r\rangle = (ns5s){}^3S_1$')
ax.set_xlabel(r'Time [$1/\Omega$]') 
ax.set_ylabel('Population') 
ax.set_ylim([0, 1])
ax.legend()

# fidelity as a function of laser linewidth
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.grid()

ax2.scatter(laser_linewidth_array/(2*pi), fidelity_errors)
ax2.set_xscale('log')
ax2.set_title('Fidelity error after $\pi$ pulse')
ax2.set_xlabel('Laser linewidth [$2\pi \cdot Hz$]')
ax2.set_ylabel('Fidelity error')

plt.show()
