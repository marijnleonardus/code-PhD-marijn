# author: Marijn Venderbosch
# 2023

import numpy as np
import matplotlib.pyplot as plt


def rabi_freq_uniformity_dependence(x, waist):
    """
    Uniformity gaussian beam across array, while intensity I/I0 goes as exp(-2x^2)
    because Rabi goes as sqrt(I) the uniformity in rabi frequency goes as exp(-x^2)
    
    inputs
    - coordinate x[m]
    - beam waist [m]
    
    output
    - uniformity (Rabi/rabi0) where rabi0 is the rabi freq. in the center of the array
    """
    rabi_uniformity = np.exp(-x**2 / waist**2)
    rabi_nonuniformity = 1 - rabi_uniformity
    return rabi_nonuniformity

max_x = 14e-6
uniformity_crit = 0.995

waist_array = 1e-6 * np.linspace(10, 200, 100 + 1)
non_uniformity_array = rabi_freq_uniformity_dependence(max_x, waist_array)

fig, ax = plt.subplots()
ax.grid()

ax.plot(waist_array / 1e-6, non_uniformity_array,
        label=r'$1-\Omega/\Omega_0$')

ax.set_yscale('log')
ax.set_xlabel('beam waist [$\mu$m]')
ax.set_ylabel('Rabi frequency error, $1-|\Omega/\Omega_0|$')

# mark point omega/omega_0 = 0.995
ax.axhline(1 - uniformity_crit, color='red', 
           label=f'$\Omega/\Omega_0={uniformity_crit}$')

ax.legend()
