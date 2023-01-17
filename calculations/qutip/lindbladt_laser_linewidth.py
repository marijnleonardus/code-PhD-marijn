# author: Marijn Venderbosch
# 2023

from qutip import (mesolve, basis,
                   sigmax, sigmay, sigmaz, 
                   ket2dm)
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np

# %% variables

rabi = 10
detuning = 0.01

# %% calculation

# define hamiltonian of spin 1/2 system with tunning rate 0.1
# decompose in pauli matrices
hamiltonian = 2 * np.pi * (rabi * qt.sigmax() + detuning * qt.sigmay())

# system starts out in |g> = (0,1) state
psi0 = qt.basis(2, 1)

# obtain density matrix
rho0 = qt.ket2dm(psi0)

# time matrix
time = np.linspace(0.0, 10.0, 100)

# solve equation taking into account loss channel
loss_channel = np.sqrt(0.05) * qt.sigmax()
result = mesolve(hamiltonian, rho0, time,
                 c_ops=loss_channel)

population_g = np.real(qt.expect(result.states, qt.projection(2, 1, 1)))
population_e = np.real(qt.expect(result.states, qt.projection(2, 0, 0)))


# %% Plotting

fig, ax = plt.subplots()
ax.grid()

#ax.plot(result.times, population_g, label=r'$\rho_{gg}$')
ax.plot(result.times, population_e, label=r'$\rho_{ee}$')
 

ax.set_xlabel('Time') 
ax.set_ylabel('Population') 
ax.legend()

plt.show() 