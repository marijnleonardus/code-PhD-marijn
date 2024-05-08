# %% imports 

import numpy as np
import matplotlib.pyplot as plt                                      

# %% variables
# polarizabilit 1S0 in atomic units. source: rydberg simulation platform. 
pol_1s0 = 274.4 + 6.4

# polarizability 3p1 in atomic units source: rydberg simulation platform. 
alpha_s_3p1 = 251.9 + 43.9
alpha_t_3p1 = 53.0
              
# formula from young, madjarov theses
theta = np.linspace(0, np.pi/2, 100)

pol_3p1_mj1 = alpha_s_3p1 + alpha_t_3p1*(3*np.cos(theta)**2-1)/2*(3*1**2-2)
pol_3p1_mj0 = alpha_s_3p1 + alpha_t_3p1*(3*np.cos(theta)**2-1)/2*(3*0**2-2)

# %% plotting
fig, ax = plt.subplots(figsize=(6, 4))

# plot 1s0, make matrix of length theta
pol_1s0 = np.linspace(pol_1s0, pol_1s0, len(theta))
ax.plot(theta, pol_1s0, label=r'${}^1S_0$')

# plot 3p1(mj=0, mj=+/-1)
ax.plot(theta, pol_3p1_mj0, label=r'${}^3P_1 \ (m_j=0)$')
ax.plot(theta, pol_3p1_mj1, label=r'${}^3P_1 \ (m_j=\pm 1$)')

ax.set_xlabel(r'$\theta$ (rad)')
ax.set_ylabel(r'$\alpha$ (atomic units)')
ax.legend()
plt.show()