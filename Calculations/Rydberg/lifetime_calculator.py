# author: Marijn Venderbosch
# January 2023

from arc import Strontium88
import numpy as np
import matplotlib.pyplot as plt

sr88 = Strontium88()

# %% variables

n_lower = int(40)
n_higher = int(70)
n_span = int(n_higher - n_lower)
n_array = np.linspace(n_lower, n_higher, n_span + 1)

# %% compute natural linewidths or Einstein coefficients

linewidth_list = []

for n in n_array:
    linewidth = sr88.getTransitionRate(n, 0, 1,  # (ns5s)3s1
                                       5, 1, 0,  # (5s5s)3p0
                                       temperature=0,
                                       s=1)
    linewidth_list.append(linewidth)

linewidth_array = np.array(linewidth_list)
linewidth_omega = linewidth_array / (2 * np.pi)
    
# %% plot result

#plt.style.use('default')
fig, ax = plt.subplots()

ax.grid()
ax.plot(n_array, linewidth_omega/1e3, label = '${}^3P_0-(ns5s)^3S_1$')
ax.set_xlabel('$n$')
ax.set_ylabel('Natural linewidth [kHz $\cdot 2\pi$]')
ax.legend()
    