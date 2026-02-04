# author marijn Venderbosch
# january 2026

"""first run histogram_and_threshold.py """
#%%

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
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
from single_atoms_class import SingleAtoms
from plot_utils import Plotting
from units import us, ms
from statistics_utils import Stats

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

#%%

# variables
processed_data_path = 'output/processed_data/roi_counts/'
rid = 'scan193823' # rydberg rabi single
rid = 'scan001915' # clock rabi
path = processed_data_path + rid + '/'
detection_threshold_file_name = os.path.join(path, "detection_threshold.npy")
binary_threshold = np.load(detection_threshold_file_name)

# load x values 
df = pd.read_csv(os.path.join(path, 'log.csv'))

## statistics
# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, path)
x_grid, surv_matrix = SingleAtomsStats.reshape_survival_matrix(df)
print("sorted data: nr ROIs, nr x_values, nr_avg: ", np.shape(surv_matrix))

# calculate mean and standard error per ROI and globally
statistics_matrix = SingleAtomsStats.calculate_survival_statistics(df) 
surv_prob_per_roi, surv_prob_global, sem_per_roi, global_sem = statistics_matrix

# calculate mean and standard error per ROI and globally
statistics_matrix = SingleAtomsStats.calculate_survival_statistics(df) 
surv_prob_per_roi, surv_prob_global, sem_per_roi, global_sem = statistics_matrix
print(surv_prob_per_roi)
# construct RoI geometry, will also work for missing ROI
# Example: If the 4th spot in the 2nd row is missing do: 1,3
#missing_row, missing_col = 4, 0
missing_row, missing_col = None, None

nr_rows, nr_cols = np.loadtxt(path + "roi_geometry.csv", dtype=int)

# grid of ROI indices. Initialize with -1 (meaning "no ROI detected here")
roi_grid = np.full((nr_rows, nr_cols), -1, dtype=int)

# Fill the grid sequentially with the detected ROI indices 
current_roi = 0
nr_rois = len(surv_prob_per_roi)
for r in range(nr_rows):
    for c in range(nr_cols):
        if r == missing_row and c == missing_col:
            continue  # Skip this spot in the grid
        if current_roi < nr_rois:
            roi_grid[r, c] = current_roi
            current_roi += 1

# plot individual Rois
fig1, ax1 = plt.subplots(nrows=nr_rows, ncols=nr_cols, 
    figsize=(2*nr_cols, 2*nr_rows),
    sharex=True, sharey=True)

axs = np.atleast_1d(ax1).ravel()

# plot individual ROIs, if missing, don't plot
for r in range(nr_rows):
    for c in range(nr_cols):
        ax = ax1[r, c]
        roi_idx = roi_grid[r, c]
        
        if roi_idx != -1: # Only plot if an ROI exists here
            ax.errorbar(x_grid/us, surv_prob_per_roi[roi_idx, :], 
                yerr=sem_per_roi[roi_idx, :], fmt='o', markersize=1.5)
            ax.set_title(f"ROI {roi_idx}")
        else:
            ax.set_title("Empty", color='red')
        #ax.set_xlim(0, 5.0)

fig1, ax1 = plt.subplots(1, ncols=nr_cols, 
    figsize=(12, 2.5), sharey=True)
## column averaging
# Choose which column to average
for target_col in range(nr_cols):
    # Get all valid ROI indices for this column from our grid
    # This logic assumes ROI 0, 1, 2, 3, 4 are Row 0
    # and ROI 5, 6, 7... are Row 1
    # Therefore, any ROI where (index % nr_cols) == target_col belongs to that column
    col_indices = [i for i in range(len(surv_prob_per_roi)) if i % nr_cols == target_col]

    if col_indices:
        roi_column = np.mean([surv_prob_per_roi[i] for i in col_indices], axis=0)
        sem_column = Stats.propagate_standard_error([sem_per_roi[i] for i in col_indices])

    ax = ax1[target_col]
    ax.plot(x_grid/ms, roi_column)
        #yerr=sem_column, fmt='o', markersize=4)
fig1.supylabel("Survival Probability")
fig1.supxlabel("Time (us)")


plt.show() 
