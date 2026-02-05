# author marijn Venderbosch
# january 2026

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# User defined libraries
from modules.roi_analysis_class import SurvivalAnalysis
from modules.fitting_functions_class import FittingFunctions, EstimateFit
from utils.statistics_utils import Stats
from utils.units import us, MHz

os.system('cls' if os.name == 'nt' else 'clear')

# raw data and processed data locations
rid = 'scan184404' # rydberg ramsey
raw_path = 'Z:\\Strontium\\Images\\2026-01-29\\'
processed_root = 'output/processed_data/'

# ROI Analysis Settings (Only needed if running analysis from scratch)
roi_config = {
    'radius': 2,
    'log_thresh': 8,
    'index_tolerance': 5
}
hist_config = {
    'nr_bins_roi': 15,
    'nr_bins_avg': 50,
    'plot_only_initial': True
}

# ROI geometry. Missing row/cols starting at 0 (python index notation)
geometry = (5, 5) # rows, cols
missing_spots = [(4, 0)]

# obtain processed survival probability data
x_grid, glob_surv, glob_surv_sem, roi_surv, roi_sem, df = SurvivalAnalysis.get_survival_data(
    rid, raw_path, processed_root, roi_config, hist_config, geometry)

# grid of ROI indices. Initialize with -1 (meaning "no ROI detected here")
roi_grid = SurvivalAnalysis.map_rois_to_grid(geometry, missing_coords=missing_spots)

# plot individual ROIs, if missing, don't plot
nr_rows = geometry[0]
nr_cols = geometry[1]
fig1, ax1 = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=(2*nr_cols, 2*nr_rows), sharex=True, sharey=True)
axs = np.atleast_1d(ax1).ravel()
for r in range(nr_cols):
    for c in range(geometry[1]):
        ax = ax1[r, c]
        roi_idx = roi_grid[r, c]
        if roi_idx != -1: # Valid ROI
            ax.errorbar(x_grid/us, roi_surv[roi_idx, :], yerr=roi_sem[roi_idx, :], fmt='o', markersize=5, alpha=0.5)
            ax.set_title(f"ROI {roi_idx}")
        else: 
            ax.set_title("Empty", color='red')

fig2, ax2 = plt.subplots()
ax2.errorbar(x_grid/us, glob_surv, yerr=glob_surv_sem, fmt='o', markersize=5, alpha=0.5)

plt.show()
