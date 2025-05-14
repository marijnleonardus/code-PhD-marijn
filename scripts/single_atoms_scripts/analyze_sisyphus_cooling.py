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
sys.path.append(modules_dir)

# user defined libraries
from single_atoms_class import SingleAtoms
from plotting_class import Plotting
from units import MHz

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-05-12\\scan171709\\'
file_name_suffix = 'image'  # import files ending with image.tif
binary_threshold = np.load(images_path + 'detection_threshold.npy')

# load x values 
df = pd.read_csv(images_path + 'log.csv')

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

fig1, ax1 = plt.subplots()
ax1.errorbar(x_grid/MHz, glob_surv, yerr=glob_surv_sem, fmt='o', color='blue')
ax1.set_xlabel('Sisyphus cooling detuning. [MHz]')
ax1.set_ylabel('Survival probability')
ax1.set_xlim([-3.65, -1.65])
#ax1.legend()

Plotting().savefig('output//','sis_cooling_surv.pdf') 
