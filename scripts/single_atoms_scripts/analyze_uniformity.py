# author marijn Venderbosch
# April 2025

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem

# user defined libraries
from modules.fitting_functions_class import FittingFunctions
from modules.image_analysis_class import ImageStats
from modules.roi_analysis_class import SurvivalAnalysis
from utils.plot_utils import Plotting
from utils.units import MHz

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
rid = 'scan174612'
raw_path = 'T:\\KAT1\\Marijn\\'
processed_root = 'output/processed_data/'

# ROI Analysis Settings (Only needed if running analysis from scratch)
roi_config = {
    'radius': 1,
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
missing_row, missing_col = None, None

# obtain processed survival probability data
x_grid, glob_surv, glob_surv_sem, roi_surv, roi_sem, df = SurvivalAnalysis.get_survival_data(
    rid, raw_path, processed_root, roi_config, hist_config, geometry)

# plot for each ROI the survival plobability as function of detuning and fit with Gaussian
nr_rois = np.shape(roi_surv)[0]
fig1, axs = plt.subplots(figsize = (7, 6), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))

# needs to be 1d to iterate. prep intial guess for fitting
axs = axs.ravel()
initial_guess = [1, -0.5, 3.5e6, 200e3] #  offset, amplitude, middle, width
popt_list = []
pcov_list = []

# x axis with more values for fit plot
x_axis_fit = np.linspace(x_grid[0], x_grid[-1], 500)

for roi_idx in range(nr_rois):
    print(nr_rois)
    axs[roi_idx].errorbar(x_grid/MHz, roi_surv[roi_idx, :], roi_sem[roi_idx, :],
        fmt='o', ms=2, capsize=1, capthick=1)
    axs[roi_idx].set_title(f'ROI {roi_idx}')

    # fit datapoints and plot result
    popt, pcov = curve_fit(FittingFunctions.gaussian_function, x_grid, roi_surv[roi_idx, :], p0=initial_guess)
    popt_list.append(popt)
    pcov_list.append(pcov)
    axs[roi_idx].plot(x_axis_fit/MHz, FittingFunctions.gaussian_function(x_axis_fit, *popt), color='red')

fig1.supxlabel('Detuning [MHz]')
fig1.supylabel('Survival probabiility')

# calculate uniformity
detunings = np.array([arr[2] for arr in popt_list])
avg_detuning = np.mean(detunings)
sem_detuning = sem(detunings)
std_deviation_detuning = np.std(detunings)
uniformity = ImageStats.calculate_uniformity(detunings)

print("Fit position:  ", np.round(avg_detuning/MHz, 2), "p/m ", np.round(sem_detuning/MHz, 2), "MHz")
print("Standard deviation: ", 100*np.round(std_deviation_detuning/avg_detuning, 3), "%")
print("uniformity: ", np.round(uniformity, 3))

Plot = Plotting('output')
Plot.savefig('uniformity_plot.png')
plt.show()

# save results to be used by `tweezers_charact_combined.py`
export_data = {
    'detunings': detunings,
    'surv_prob': roi_surv,
    'sem_surv_prob': roi_sem,
    'x_grid': x_grid,
    'x_axis_fit': x_axis_fit,
    'popt_list': popt_list
}

np.savez('output/combined_figs/uniformity_data_dict.npz', **export_data)
