
import numpy as np
import matplotlib.pyplot as plt                                      

# polarizabilit 1S0 in atomic units. source: rydberg simulation platform. 
pol_1s0 = 274.4 + 6.4

# polarizability 3p1 in atomic units source: rydberg simulation platform. 
alpha_s_3p1 = 251.9 + 43.9
alpha_t_3p1 = 53.0
              
# formula from young, madjarov theses
theta_rad = np.linspace(0, np.pi/2, 100)
theta_deg = theta_rad*180/np.pi

pol_3p1_mj1 = alpha_s_3p1 + alpha_t_3p1*(3*np.cos(theta_rad)**2-1)/2*(3*1**2-2)
pol_3p1_mj0 = alpha_s_3p1 + alpha_t_3p1*(3*np.cos(theta_rad)**2-1)/2*(3*0**2-2)

fig, ax = plt.subplots(figsize=(4, 3))
pol_1s0 = np.linspace(pol_1s0, pol_1s0, len(theta_rad))
ax.plot(theta_deg, pol_1s0, label=r'${}^1S_0$', color='orange')
ax.plot(theta_deg, pol_3p1_mj0, label=r'${}^3P_1 \ (m_{\text{j}}=0)$', color='blue')
ax.plot(theta_deg, pol_3p1_mj1, label=r'${}^3P_1 \ (m_{\text{j}}=\pm 1$)', color='navy')
ax.set_xlabel(r'$\theta$ (deg.)')
ax.set_ylabel(r'$\alpha_{\text{tot}}$ (atomic units)')
ax.legend()

plt.savefig('output/polarizability_3p1.pdf', bbox_inches='tight', dpi=400)
plt.show()