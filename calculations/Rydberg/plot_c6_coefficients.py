# author: Marijn Venderbosch
# january 2023


import numpy as np
from classes.atomic_data_functions import calculate_c6_coefficients
import matplotlib.pyplot as plt
import matplotlib


# %% variables

n_start = 40
n_end = 80
n_numbers = n_end - n_start + 1

n_array = np.linspace(n_start, n_end, n_numbers, dtype=np.int64)


# %% compute C6 coefficients

c6_list = []

for n in n_array:
    # get C6 coefficient
    c6_energy = calculate_c6_coefficients(n, 0, 1, 0)
    
    # divide by n^11 with intermediate step to avoid integer overlow
    # where the number is too large to store in normal int64 64 bit number
    n_float = float(n)
    n_term = n_float**11
    c6_coefficient = c6_energy / n_term
    
    # store result
    c6_list.append(c6_coefficient)
    
c6_array = np.array(c6_list)


# %% plotting

matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

fig, ax = plt.subplots()
ax.scatter(n_array, c6_array[:,0])

ax.set_xlabel('$n$')
ax.set_ylabel('$C_6 \cdot n^{11}$ coefficients [atomic units]')

plt.show()

