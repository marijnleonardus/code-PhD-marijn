# author marijn Venderbosch
# january 2026

"""first run histogram_and_threshold.py """
#%%

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
import pandas as pd

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from fitting_functions_class import FittingFunctions
from image_analysis_class import ImageStats
from single_atoms_class import SingleAtoms
from plot_utils import Plotting
from units import us

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

#%%

# variables
processed_data_path = 'output/processed_data/roi_counts/'
rid = 'scan193823'
path = processed_data_path + rid + '/'
detection_threshold_file_name = os.path.join(path, "detection_threshold.npy")
binary_threshold = np.load(detection_threshold_file_name)

# load x values 
df = pd.read_csv(os.path.join(path, 'log.csv'))

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, path)
x_grid, surv_matrix = SingleAtomsStats.reshape_survival_matrix(df)
print("sorted data: nr ROIs, nr x_values, nr_avg: ", np.shape(surv_matrix))

# calculate mean and standard error per ROI and globally
statistics_matrix = SingleAtomsStats.calculate_survival_statistics(df) 
surv_prob_per_roi, surv_prob_global, sem_per_roi, global_sem = statistics_matrix
print(surv_prob_per_roi.shape, surv_prob_global.shape, sem_per_roi.shape, global_sem.shape)

fig, ax = plt.subplots(figsize=(4, 1.5))
for roi_idx in range(np.shape(surv_prob_per_roi)[0]):
    ax.errorbar(x_grid/us, surv_prob_per_roi[roi_idx, :], yerr=sem_per_roi[roi_idx, :],
    fmt='o', markersize=1.5, color='blue')
ax.set_xlabel('us')
ax.set_ylabel('surv. prob.')
ax.set_xlim(0, 3.0)

fig2, ax2 = plt.subplots(figsize=(3, 2))
ax2.errorbar(x_grid/us, surv_prob_global, global_sem, fmt='o', markersize=1.5, color='blue')
ax2.set_xlabel('us')
ax2.set_ylabel('global survival')
ax2.set_xlim(0, 1.5)

plt.show()
