# author: Marijn Venderbosch
# january 2023

import numpy as np
from functions.atomic_data_functions import calculate_c6_coefficients
import matplotlib.pyplot as plt
import matplotlib
from functions.fitting_functions import fit_n11_dependence
from scipy.optimize import curve_fit


# %% variables

n_start = 40
n_end = 80
n_numbers = n_end - n_start + 1

n_array = np.linspace(n_start, n_end, n_numbers, dtype=np.int64)


# %% compute C6 coefficients

c6_overn11_list = []
c6_list = []

for n in n_array:
    # get C6 coefficient
    c6 = calculate_c6_coefficients(n, 0, 1, 0)
    c6_list.append(c6)
    
    # divide by n^11 with intermediate step to avoid integer overlow
    # where the number is too large to store in normal int64 64 bit number
    n_float = float(n)
    n_term = n_float**11
    c6_coefficient_overn11 = c6 / n_term
    
    # store result, with and without divide by n^11
    c6_overn11_list.append(c6_coefficient_overn11)

c6_overn11_array = np.array(c6_overn11_list)
c6_array = np.array(c6_list)

# %% fit data

# =============================================================================
# popt, _ = curve_fit(fit_n11_dependence, n_array, abs(c6_array[:,0]), p0=[0, 10**8])
# c6_fit_result = fit_n11_dependence(n_array, *popt)
# 
# =============================================================================
# %% plotting

matplotlib.rcParams['font.family'] = 'sansserif'
matplotlib.style.use('default')

fig, ax = plt.subplots()
ax.grid()

ax.scatter(n_array, c6_overn11_array[:,0])

ax.set_xlabel('$n$')
ax.set_ylabel('$C_6 / n^{11}$ coefficients [atomic units]')

fig2, ax2= plt.subplots()

ax2.grid()
ax2.plot(n_array, abs(c6_array[:,0]))
# ax2.plot(n_array, c6_fit_result)

ax2.set_yscale('log')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$|C_6|$ [GHz $\mu$m$^6$]')

plt.show()

