# author: Marijn Venderbosch
# April 2025

import numpy as np
import os
import matplotlib.pyplot as plt

# User defined libraries
from modules.roi_analysis_class import SurvivalAnalysis
from utils.units import MHz, pol_1s0, pol_3p1_mj1
from utils.plot_utils import Plotting

os.system('cls' if os.name == 'nt' else 'clear')

# raw data and processed data locations
rid = 'scan171709'
raw_path = 'Z:\\Strontium\\Images\\2025-05-12\\'
processed_root = 'output/processed_data'

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
geometry = (5, 5) # rows, cols

# Experiment Physics Settings
power_tweezers = 300
power_tweezers_max = 340
trapdepth = 11.08*MHz*power_tweezers/power_tweezers_max

# obtain processed survival probability data
x_grid, glob_surv, glob_surv_sem, roi_surv, roi_sem, df = SurvivalAnalysis.get_survival_data(
    rid, raw_path, processed_root, roi_config, hist_config, geometry)

# Find Best Cooling Frequency
max_idx = np.argmax(glob_surv)
max_surv_prob = np.round(glob_surv[max_idx], 5)
max_surv_prob_err = np.round(glob_surv_sem[max_idx], 5)
best_cooling_freq = x_grid[max_idx]
print(f"Best survival: {max_surv_prob} ± {max_surv_prob_err} at {best_cooling_freq/MHz:.3f} MHz")

# plotting
fig_width = 2.5 
fig_height = 1.5
fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
ax1.errorbar(x_grid/MHz, glob_surv, yerr=glob_surv_sem, ms=1.5, fmt='o', color='blue', label='Data')
diff_ac_stark = trapdepth*(1 - pol_3p1_mj1/pol_1s0)
ax1.axvline(x=diff_ac_stark/MHz, color='red', linewidth=1.5, label='AC Stark Shift')
ax1.set_xlabel('Detuning (MHz)')
ax1.set_ylabel('Avg. survival probability')
# ax1.set_xlim([-3.65, -1.65]) # Uncomment if needed
# ax1.legend(fontsize=6)
print(f"AC Stark Shift: {diff_ac_stark/MHz:.3f} MHz")

Plot = Plotting('output')
Plot.savefig('sis_cooling_surv.pdf') 
plt.show()
