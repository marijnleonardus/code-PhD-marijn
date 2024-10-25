# author: Marijn Venderbosch
# 2023

from qutip import mesolve, sigmax, basis, sigmay, sigmaz
import matplotlib.pyplot as plt
import numpy as np

# %% Qutip

# define hamiltonian of spin 1/2 system with tunning rate 0.1
# sigmax is the pauli spin matrix
H = 2 * np.pi * 0.1 * sigmax()

# system starts out in (1,0) (up) state
psi0 = basis(2, 0)

# time matrix
times = np.linspace(0.0, 10.0, 100)

# solve equation taking into account loss channel
loss_channel = np.sqrt(0.05) * sigmax()
result = mesolve(H, psi0, times, [loss_channel], [sigmaz(), sigmay()])
# obtain expectation values
result.expect


# %% Plotting

fig, ax = plt.subplots()

ax.plot(result.times, result.expect[0]) 
ax.plot(result.times, result.expect[1]) 

ax.set_xlabel('Time') 
ax.set_ylabel('Expectation values') 
ax.legend(("Sigma-Z", "Sigma-Y")) 

plt.show() 