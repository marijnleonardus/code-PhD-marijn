# author: Marijn Venderbosch
# 2023

import qutip as qt
import matplotlib.pyplot as plt
import numpy as np

# %% variables

rabi = 2 * np.pi * 1
detuning = 2 * np.pi * 0.1

# rydberg inverse lifetime
gamma_r = 2 * np.pi * 0.01

# linewidth
linewidth = 2 * np.pi * (1e3/1e6)

# %% calculation

# define hamiltonian of spin 1/2 system with tunning rate 0.1
# decompose in pauli matrices
hamiltonian = 0.5 * (rabi * qt.sigmax() + - detuning * qt.sigmaz())

# system starts out in |g> = (0,1) state
psi0 = qt.basis(2, 1)

# obtain density matrix
rho0 = qt.ket2dm(psi0)

# time matrix
time = np.linspace(0.0, 50.0, 1000)

# solve equation taking into account loss channel
lindbladt_spont_em = gamma_r * (qt.sigmaz() - 0.5 * qt.sigmax())

lindbladt_linewidth = -linewidth * qt.sigmax()

lindbladt_total = lindbladt_spont_em + lindbladt_linewidth

result = qt.mesolve(hamiltonian, rho0, time,
                   c_ops=lindbladt_total)

population_g = np.real(qt.expect(result.states, qt.projection(2, 1, 1)))
population_e = np.real(qt.expect(result.states, qt.projection(2, 0, 0)))
coherence = np.real(qt.expect(result.states, qt.projection(2, 0, 1)))


# %% Plotting

fig, ax = plt.subplots()
ax.grid()

#ax.plot(result.times, population_g, label=r'$\rho_{gg}$')
ax.plot(result.times, population_e, label=r'$\rho_{ee}$')
#ax.plot(result.times, coherence, label=r'$\rho_{eg}$')

 

ax.set_xlabel('Time [qutip units]') 
ax.set_ylabel('Population') 
ax.set_ylim([0, 1])
ax.legend()

plt.show() 