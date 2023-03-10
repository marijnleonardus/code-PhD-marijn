# author: Marijn Venderbosch
# january 2023

# needed for vscode
import sys
import os
sys.path.append(os.getcwd())

from arc import Strontium88
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

atom = Strontium88()

# %% variables

# number of principal quantum numbers to consider
start_n = 5
end_n = 40
number_n = end_n - start_n + 1

n_array = np.linspace(5, 40, 40-5+1)

defect_list = []

for n in n_array:
    # n, l, j. s=1 because triplet
    defect = atom.getQuantumDefect(n, 0, 1, s=1)
    defect_list.append(defect)
    
defect_array = np.array(defect_list)

# %% plotting

fig, ax = plt.subplots()

matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

ax.grid()
ax.scatter(n_array, defect_array)
ax.set_xlabel('$n$')
ax.set_ylabel(r'$\delta_{n}$')

plt.show()
