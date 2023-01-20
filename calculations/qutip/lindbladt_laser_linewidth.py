#!/usr/bin/env python

# author: Marijn Venderbosch
# 2023

import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
from classes.plotting_class import Plotting

# %% variables

# laser
laser_rabi_frequency = 2 * np.pi * 10e6  # Hz
laser_detuning = 2 * np.pi * 0  # Hz
laser_linewidth = 2 * np.pi * 1e6 # Hz

# rydberg lifetime
rydberg_lifetime = 100e-6  # s for n=61 from Madhav paper

# time matrix
time = np.linspace(0.0, 200., 1000) 

# %% convert units

# compute decay rate from rydberg state
rydberg_decay_rate = 1 / rydberg_lifetime   # Hz

# make dimensionless
def make_dimensionless(number, rabi_freq):
    """make units dimensionless in units of rabi frequencies"""
    return number / rabi_freq

# divide by rabi frequency to make dimensionless
laser_detuning = make_dimensionless(laser_detuning, laser_rabi_frequency)
laser_linewidth= make_dimensionless(laser_linewidth, laser_rabi_frequency)
rydberg_decay_rate = make_dimensionless(rydberg_decay_rate, laser_rabi_frequency)
laser_rabi_frequency = make_dimensionless(laser_rabi_frequency, laser_rabi_frequency)

# %% qutip simulation 


def integrate_qubit_population():
    # define hamiltonian decomposed in pauli matrices
    # units are hbar=1 (atomic units)
    hamiltonian = 0.5 * (laser_rabi_frequency * qt.sigmax() + - laser_detuning * qt.sigmaz())
    
    # system starts out in |g> = (0,1) state
    psi0 = qt.basis(2, 1)
    
    # obtain density matrix
    rho0 = qt.ket2dm(psi0)

    # define lindblad operators and the total lindbladt superoperator
    lindblad_spont_em = rydberg_decay_rate * (qt.sigmaz() - 0.5 * qt.sigmax())
    lindblad_linewidth = laser_linewidth * qt.sigmax()
    lindblad_total = [lindblad_spont_em, lindblad_linewidth]
    
    # solve master equation given hamiltonian, initial state rho0, 
    # and collapse operators c_ops
    result = qt.mesolve(hamiltonian, rho0, time, c_ops=lindblad_total)
    return result

result = integrate_qubit_population()
    

# %% Plotting

population_g = np.real(qt.expect(result.states, qt.projection(2, 1, 1)))
population_r = np.real(qt.expect(result.states, qt.projection(2, 0, 0)))
coherence = np.real(qt.expect(result.states, qt.projection(2, 0, 1)))

fig, ax = plt.subplots(figsize=(4, 3))
ax.grid()

ax.plot(result.times, population_r, label=r'$\rho_{rr}$')
#ax.plot(result.times, population_g, label=r'$\rho_{gg}$')
#ax.plot(result.times, coherence, label=r'$\rho_{eg}$')

ax.set_title(r'Population of $|r\rangle = (61sns){}^3S_1$')
ax.set_xlabel(r'Time [$2\pi/ \Omega$]') 
ax.set_ylabel('Population') 
ax.set_ylim([0, 1])
ax.legend()

Plotting.saving('calculations/qutip/output/',
                'population_vs_time.png')

plt.show() 
