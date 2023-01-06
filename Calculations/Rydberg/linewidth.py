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

# %% functions

def compute_natural_linewidths(array, l1, j1,
                               n2, l2, j2):
    
    """computes natural linewidth using ARC diatomic function"""
    
    # init empty list to store result
    linewidths_list = []
    
    # iterate over n
    for n in array:
        linewidth = sr88.getTransitionRate(n, l1, j1,  
                                           n2, l2, j2,  
                                           temperature=0,
                                           s=1)
        linewidths_list.append(linewidth)
    
    # store as np array
    linewidths_array = np.array(linewidths_list)
    return linewidths_array

# %% compute 

linewidths_3p0 = compute_natural_linewidths(n_array, 0, 1, 
                                            5, 1, 0)

    
# %% plot result

#plt.style.use('default')

# plot linewidhts
fig, ax = plt.subplots()

ax.grid()

ax.plot(n_array, linewidths_3p0 / 2 / np.pi, label = '${}^3P_0-(ns5s)^3S_1$')

ax.set_xlabel('$n$')
ax.set_ylabel('Natural linewidth [Hz $\cdot 2\pi$]')
ax.legend()
    