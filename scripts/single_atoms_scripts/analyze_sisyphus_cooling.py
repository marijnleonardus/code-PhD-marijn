# author marijn Venderbosch
# April 2025

"""first run histogram_and_threshold.py """


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
utils_dir = os.path.abspath(os.path.join(script_dir, '../../utils'))
sys.path.append(modules_dir)
sys.path.append(utils_dir)

# user defined libraries
from single_atoms_class import SingleAtoms
from units import MHz, pol_1s0, pol_3p1_mj1
from plot_utils import Plotting

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-05-12\\scan171709\\'
file_name_suffix = 'image'  # import files ending with image.tif
binary_threshold = np.load(images_path + 'detection_threshold.npy')

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# settings experiment
power_tweezers = 300
power_tweezers_max = 340
trapdepth = 11.08*MHz*power_tweezers/power_tweezers_max

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_grid, surv_matrix = SingleAtomsStats.reshape_survival_matrix(df)
print("sorted data: nr ROIs, nr x_values, nr_avg: ", np.shape(surv_matrix))

# calculate mean and standard error per ROI and globally
statistics_matrix = SingleAtomsStats.calculate_survival_statistics(df) 
glob_surv = statistics_matrix[1]
glob_surv_sem = statistics_matrix[3]

max_idx = np.argmax(glob_surv)
max_surv_prob = np.round(glob_surv[max_idx], 4)
max_surv_prob_err = np.round(glob_surv_sem[max_idx], 3)
best_cooling_freq = x_grid[max_idx]

print("best survival: ", max_surv_prob, " pm ", max_surv_prob_err, " at ", best_cooling_freq/MHz, " MHz")

fig_width = 2.5  # inches, matches one column
fig_height = 1.5
fig1, ax1 = plt.subplots(figsize = (fig_width, fig_height))
ax1.errorbar(x_grid/MHz, glob_surv, yerr=glob_surv_sem, ms=1.5, fmt='o', color='blue')
ax1.set_xlabel('Detuning (MHz))')
ax1.set_ylabel('Avg. survival probability')
ax1.set_xlim([-3.65, -1.65])

#ax1.axvline(x=best_cooling_freq/MHz, color='red', linestyle='dashed', linewidth=1.5)

# add line for AC Stark shifted resonance
diff_ac_stark = trapdepth*(1 - pol_3p1_mj1/pol_1s0)
ax1.axvline(x=diff_ac_stark/MHz, color='red', linewidth=1.5)

Plot=Plotting('output')
Plot.savefig('sis_cooling_surv.pdf') 
print(diff_ac_stark)
plt.show()
