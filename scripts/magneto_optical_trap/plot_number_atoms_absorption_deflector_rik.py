import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils', '../../raw_data/deflector'])

from plot_utils import Plotting

# load data
number_atoms_array = np.loadtxt('raw_data/deflector/number_atoms_array.npy')
errors_number_atoms = np.loadtxt('raw_data/deflector/errors_number_atoms.npy')
power = np.loadtxt('raw_data/deflector/power_array.npy')

# plotting
fig_width = 3.375*0.5 # inches, matches one column
fig_height = fig_width*0.61

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.errorbar(power, number_atoms_array/1e6, yerr=errors_number_atoms/1e6, ms=1.5, fmt='o', color='blue')
ax.set_ylabel(r"$N\,(\times 10^6)$")
ax.set_xlabel(r'$P_{\mathrm{defl}}\,(\mathrm{mW})$')

Plot = Plotting('output')
Plot.savefig("number_atoms_deflector.pdf")
plt.show()
