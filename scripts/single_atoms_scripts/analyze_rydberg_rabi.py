# author marijn Venderbosch
# january 2026

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# User defined libraries
from modules.roi_analysis_class import SurvivalAnalysis
from modules.fitting_functions_class import FittingFunctions
from utils.statistics_utils import Stats
from utils.units import us

os.system('cls' if os.name == 'nt' else 'clear')

# raw data and processed data locations
#rid = 'scan193823' # rydberg rabi single
rid = 'Scan172435'
#rid = 'scan001915' # clock rabi
raw_path = 'Z:\\Strontium\\Images\\2026-01-28\\'
processed_root = 'output/processed_data/'

# ROI Analysis Settings (Only needed if running analysis from scratch)
roi_config = {
    'radius': 2,
    'log_thresh': 10,
    'index_tolerance': 5
}
hist_config = {
    'nr_bins_roi': 15,
    'nr_bins_avg': 50,
    'plot_only_initial': True
}

# ROI geometry. If the 4th c in the 2nd row is missing do: 1,3
geometry = (5, 5) # rows, cols
missing_row, missing_col = 4, 0

# obtain processed survival probability data
x_grid, glob_surv, glob_surv_sem, roi_surv, roi_sem, df = SurvivalAnalysis.get_survival_data(
    rid, raw_path, processed_root, roi_config, hist_config, geometry)

# grid of ROI indices. Initialize with -1 (meaning "no ROI detected here")
roi_grid = SurvivalAnalysis.map_rois_to_grid(geometry, [(missing_row, missing_col)])

# plot individual ROIs, if missing, don't plot
nr_rows = geometry[0]
nr_cols = geometry[1]
t_us = x_grid/us
fig1, ax1 = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(2*nr_cols, 2*nr_rows), sharex=True, sharey=True)
axs = np.atleast_1d(ax1).ravel()
for r in range(nr_cols):
    for c in range(geometry[1]):
        ax = ax1[r, c]
        roi_idx = roi_grid[r, c]
        if roi_idx != -1: # Valid ROI
            ax.errorbar(t_us, roi_surv[roi_idx, :], yerr=roi_sem[roi_idx, :], fmt='o', markersize=5, alpha=0.5)
            ax.set_title(f"ROI {roi_idx}")
        else: 
            ax.set_title("Empty", color='red')

# fit parameters []
init_guess = [0.5  ,2.5    ,0.5, 0.8,     0.6] # amplitude, damping time, frequency, phase, offset
lower_b =    [0.05  ,0.3    ,0.3, -np.pi,0.2]
upper_b =    [0.5   ,30.0  ,3, np.pi ,0.9]

# column averaging
fig2, ax2 = plt.subplots(nrows=nr_rows, ncols=1, figsize=(nr_cols, 2*nr_cols), sharex=True, sharey=True)

# Create a high-resolution time axis for the fit curve
t_fit = np.linspace(t_us[0], t_us[-1], 1000)

for target_col in range(nr_cols):
    # Get all valid ROI indices for this column
    col_indices = [i for i in range(len(roi_surv)) if i % nr_cols == target_col]
    
    if col_indices:
        # Extract the relevant means and SEMs for this column
        current_means = [roi_surv[i] for i in col_indices]
        current_sems = [roi_sem[i] for i in col_indices]
        
        # Calculate weighted average and propagated SE, and plot result
        roi_column, sem_column = Stats.weighted_average_and_se(current_means, current_sems)

        # plot datapoints
        ax = ax2[target_col]
        ax.errorbar(t_us, roi_column, yerr=sem_column, fmt='o', markersize=5, alpha=0.5)
        ax.set_title(f"Column {target_col}")

        # fit data. [amplitude, damping time, frequency, phase, offset]
        try:
            popt, pcov = curve_fit(FittingFunctions.damped_sin_wave, 
                t_us, roi_column, p0=init_guess, sigma=sem_column, bounds=(lower_b, upper_b), maxfev=10000)
            y_fit = FittingFunctions.damped_sin_wave(t_fit, *popt)
            ax.plot(t_fit, y_fit, color='red', linestyle='-', linewidth=2, label=f'Fit: {popt[2]:.2f} MHz')
            ax.set_xlim(0, 3)
            print(popt)
        except RuntimeError:
            print(f"Failed to fit column {target_col}")
            continue

fig2.supylabel("Survival Probability")
fig2.supxlabel("Time (us)")
plt.tight_layout()
plt.show()
