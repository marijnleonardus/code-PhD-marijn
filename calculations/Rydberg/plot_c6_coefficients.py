# author: Marijn Venderbosch
# january 2023

from arc import PairStateInteractions, Strontium88
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from classes.conversion_functions import eh

# %% variables

n_start = 40
n_end = 80
n_numbers = n_end - n_start + 1

n_array = np.linspace(n_start, n_end, n_numbers)
n_array = n_array.astype(int)

def calculate_c6_coefficients(n, l, j, mj):
    """calculate C6 coefficients using ARC library
    
    parameters
    -------------
    n: integer
        principal quantum number 
    l: integer:
        angular momentum quantum number
    j: integer: 
        total angular momentum quantum number
    mj: integer:
        secondary total angular momentum quantum number
    
    Assumes the quantum numbers are identical for the 2 atoms
    
    returns
    ----------------
    c6: float
        van der waals interaction coefficient
        
    example
    -------------
    So for (61s5s) 3P0 mj=0 state of Sr88:
    - 61, 0, 1, 0, 1
    """
    
    calc = PairStateInteractions(
                                 Strontium88(),
                                 n, l, j,
                                 n, l, j,
                                 mj, mj,
                                 s=1
                                 )
    theta = 0
    phi = 0
    deltaN = 5
    deltaE = 30e9 # in Hz
    
    c6, eigenvectors = calc.getC6perturbatively(theta, phi, 5,
                                                deltaE,
                                                degeneratePerturbation=True)
    # getC6perturbatively returns the C6 coefficients
    # expressed in units of GHz mum^6.
    # Conversion to atomic units:
    c6 = c6 / eh
    # These results should still be divided by n^{11}
    # to be plotted as in Fig. 2(c).
    return c6

c6_list = []

for n in n_array:
    c6_energy = calculate_c6_coefficients(n, 0, 1, 0)
    
    c6_coefficient = c6_energy / n**(11)
    c6_list.append(c6_coefficient)
    
c6_array = np.array(c6_list)

# %% plotting

fig, ax = plt.subplots()
ax.plot(n_array, abs(c6_array[:,0]))
ax.set_yscale('log')

ax.set_xlabel('$n$')
ax.set_ylabel('$C_6$ coefficients [atomic units]')

plt.show()

