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
add_local_paths(__file__, ['../../modules', '../../utils'])

from plot_utils import Plotting

# load data
number_atoms_array = np.loadtxt('output/processed_data/number_atoms_array.npy')
errors_number_atoms = np.loadtxt('output/processed_data/errors_number_atoms.npy')
power = np.loadtxt('output/processed_data/power_array.npy')

# plotting
fig_width = 3.375  # inches, matches one column
fig_height = fig_width*0.61

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.errorbar(power, number_atoms_array, yerr=errors_number_atoms, ms=2, fmt='o', color='blue')
ax.set_ylabel("Number of atoms $N$")
ax.set_xlabel("Deflector Beam Power [mW]")

Plot = Plotting('output')
Plot.savefig("number_atoms_deflector.pdf")
