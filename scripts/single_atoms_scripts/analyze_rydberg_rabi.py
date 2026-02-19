# author marijn Venderbosch | Refactored January 2026
import numpy as np
import os
import matplotlib.pyplot as plt

from modules.roi_analysis_class import SurvivalAnalysis
from modules.fitting_functions_class import FittingFunctions, FitRabiOscillations
from utils.statistics_utils import Stats
from utils.units import us, MHz

os.system('cls' if os.name == 'nt' else 'clear')

# data loading
# ramsey
#rid = 'scan184404' 
#raw_path = 'Z:\\Strontium\\Images\\2026-01-29\\'

# rabi
rid = 'scan065532' #rabi individual
raw_path = 'Z:\\Strontium\\Images\\2026-02-13\\'

processed_root = 'output/processed_data/'

roi_config = {
    'radius': 3,
    'log_thresh': 10,
     'index_tolerance': 5
    }

hist_config = {
    'nr_bins_roi': 15,
    'nr_bins_avg': 50,
    'plot_only_initial': True
    }

# ROI geometry
geometry = (5, 5) 
missing_spots = [(None, None)
    ]

# physics parameters. fitting functions and parameters
decay_guess = 10 # us
fit_bounds = (# ampl, tau, rabi freq (MHz), phase, offset
    [0, 0, 0.5, -2*np.pi, 0],
    [1, 100,5, 2*np.pi, 1]
)
target_func = 'dephasing_exponential'
target_model_glob = 'dephasing_sin_gaussian'

# load data
x_grid, glob_surv, glob_sem, roi_surv, roi_sem, _ = SurvivalAnalysis.get_survival_data(
    rid, raw_path, processed_root, roi_config, hist_config, geometry)

# map ROIs to correct geometry
roi_grid = SurvivalAnalysis.map_rois_to_grid(geometry, missing_coords=missing_spots)
x_fit = np.linspace(x_grid[0], x_grid[-1], 1000)

# scale time in units of us, which improves fitting accuracy. For 1e-6 times, close to machine presicion
t_us = x_grid/us
t_us_fit = np.linspace(t_us[0], t_us[-1], 1000)

# plot individual ROIs
fig1, ax1 = plt.subplots(*geometry, figsize=(10, 10), sharex=True, sharey=True)
fit_param_grid = np.full(geometry, np.nan) 
target_func = 'full_decay'
fit_func = FittingFunctions.get_model(target_func)
for r in range(geometry[0]):
    for c in range(geometry[1]):
        ax = ax1[r, c]
        roi_idx = roi_grid[r, c]
        
        if roi_idx != -1:
            y, y_err = roi_surv[roi_idx, :], roi_sem[roi_idx, :]
            ax.errorbar(t_us, y, yerr=y_err, fmt='o', ms=4, alpha=0.4, label='Data')
            fitter = FitRabiOscillations(t_us, y, y_err, decay_guess, model=target_func)
            popt, pcov = fitter.perform_fit()
            if popt is not None:
                fit_param_grid[r, c] = popt[1] 
                ax.plot(t_us_fit, fit_func(t_us_fit, *popt), 'b-', lw=1.5)
                ax.set_title(f"ROI {roi_idx}")
            else:
                ax.set_title(f"ROI {roi_idx}: Fit Fail", color='orange')
        else:
            ax.set_title("Empty", color='gray')

fig1.supxlabel('Time [us]')
fig1.supylabel('Survival Probability')
plt.tight_layout()

# Heatmap of fitting results (rabi freq, decay time)
fig_map, ax_map = plt.subplots(figsize=(5, 4))
current_cmap = plt.cm.viridis.copy()
current_cmap.set_bad(color='white')

im = ax_map.imshow(fit_param_grid, cmap=current_cmap, interpolation='nearest')
plt.colorbar(im, ax=ax_map, label='Rabi Frequency (MHz)')

ax_map.set_title(f"Rabi Frequency Heatmap: {rid}")
ax_map.set_xticks(range(geometry[1])); ax_map.set_yticks(range(geometry[0]))
ax_map.set_xlabel("Column Index"); ax_map.set_ylabel("Row Index")

# column averaging
fig2, ax2 = plt.subplots(geometry[1], 1, figsize=(6, 10), sharex=True)
for c in range(geometry[1]):
    # Find all valid ROIs in this column
    indices = [roi_grid[r, c] for r in range(geometry[0]) if roi_grid[r, c] != -1]
    
    if indices:
        y_col, err_col = Stats.weighted_average_and_se(roi_surv[indices], roi_sem[indices])
        ax2[c].errorbar(t_us, y_col, yerr=err_col, fmt='o', ms=4, alpha=0.5)
        popt, pcov = FitRabiOscillations(t_us, y_col, err_col, decay_guess, model=target_func).perform_fit()
        if popt is not None:
            ax2[c].plot(t_us_fit, fit_func(t_us_fit, *popt), 'b-')
            ax2[c].set_title(f"Column {c}: {popt[2]:.2f} MHz")
            rabi = np.round(popt[2], 2)
            rabi_error = np.sqrt(pcov[2, 2])
            tau = np.round(popt[1], 2)
            tau_error = np.sqrt(pcov[1, 1])
            print(f"Column {c}: {popt[2]:.2f}  +/- {rabi_error:.2f} MHz, {tau:.2f} us +/- {tau_error:.2f} us")

# --- Isolated 3rd Column Analysis ---
target_col = 3  # The 3rd column (0-indexed)
fig_col3, ax_col3 = plt.subplots(figsize=(4, 3))

# Find all valid ROIs in the 3rd column
indices = [roi_grid[r, target_col] for r in range(geometry[0]) if roi_grid[r, target_col] != -1]

if indices:
    y_col, err_col = Stats.weighted_average_and_se(roi_surv[indices], roi_sem[indices])
    ax_col3.errorbar(t_us, y_col, yerr=err_col, fmt='o', ms=6, alpha=0.6)
    fitter = FitRabiOscillations(t_us, y_col, err_col, decay_guess, model=target_func)
    popt, pcov = fitter.perform_fit()
    
    if popt is not None:
        ax_col3.plot(t_us_fit, fit_func(t_us_fit, *popt), 'b-', lw=2)
ax_col3.set_xlabel(r'Rydberg pulse duration ($\mu$s)')
ax_col3.set_ylabel(r'Average survival probability')
ax_col3.legend()
plt.tight_layout()

# global averaging
fit_func_glob = FittingFunctions.get_model(target_model_glob)

mask = t_us <= 3

t_us = t_us[mask]
glob_surv = glob_surv[mask]
glob_sem = glob_sem[mask]

fig3, ax3 = plt.subplots()
ax3.errorbar(t_us, glob_surv, yerr=glob_sem, fmt='o', ms=4, alpha=0.5)
popt_glob, pcov_glob = FitRabiOscillations(t_us, glob_surv, glob_sem, decay_guess, bounds=fit_bounds, model=target_model_glob).perform_fit()
if popt_glob is not None:
    ax3.plot(t_us_fit, fit_func_glob(t_us_fit, *popt_glob), 'b-')
ax3.set_xlabel('time (us)')
ax3.set_ylabel('survival probability')
plt.tight_layout()

#print(np.round(popt_glob[1], 2), ' us')

plt.show()
