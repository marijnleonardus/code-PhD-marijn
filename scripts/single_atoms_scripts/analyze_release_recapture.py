# author marijn Venderbosch
# April 2025

"""first run histogram_and_threshold.py """

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from single_atoms_class import SingleAtoms
from plotting_class import Plotting
from utils.units import us

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-04-01\\scan104728\\'
binary_threshold = 13680
roi_radius = 1
center_weight = 3

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_grid, surv_matrix = SingleAtomsStats.reshape_survival_matrix(df)
print("sorted data: nr ROIs, nr x_values: ", np.shape(surv_matrix))

# calculate mean and standard error per ROI and globally
statistics_matrix = SingleAtomsStats.calculate_survival_statistics(df) 
surv_prob_glob = statistics_matrix[1]
surv_prob_glob_sem = statistics_matrix[3]

fig1, ax1 = plt.subplots()
ax1.errorbar(x_grid/us, surv_prob_glob, yerr=surv_prob_glob_sem,
    fmt='o', color='blue', ms=2, label='survival probability')
ax1.set_xlabel(r'Release time [$\mu$s]')
ax1.set_ylabel('Survival probabiility')

Plotting().savefig('output//','release_recapture_fit.png') 

# export data
df_survival = pd.DataFrame({
    'x_values': x_grid,
    'survival_prob': surv_prob_glob,
    'error_survival_prob': surv_prob_glob_sem
})
df_survival.to_csv('output//release_recap_raw_data.csv', index=False)
