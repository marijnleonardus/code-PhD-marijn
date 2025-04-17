# author: Marijn Venderbosch
# December 2024

"""plots fluorescence as a function of detuning for each ROI and average over all ROIs
first run `calculate_roi_counts_plot_avg.py` to get the roi_counts_matrix.npy file	
"""

 # %% 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from fitting_functions_class import FittingFunctions
from single_atoms_class import SingleAtoms
from data_handling_class import sort_raw_measurements

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-03-27\\scan095139\\'
show_plots = True
MHz = 1e6

# %% load and sort data

# load ROI counts from npy
# (nr ROIs, nr images)
roi_counts_matrix = np.load(os.path.join(images_path, 'roi_counts_matrix.npy'))
print("raw data: (nr ROIs, nr images): ", np.shape(roi_counts_matrix))

# reshape roi_counts_matrix depending on the number of averages
# laod x_values. If multiple averages used x values contains duplicates
df = pd.read_csv(images_path + 'log.csv')

# sort raw measurement data based on x values
nr_avg, x_values, roi_counts_sorted = sort_raw_measurements(df, roi_counts_matrix)

# reshape sorted data from 2D to 3D array
nr_rois = int(np.shape(roi_counts_sorted)[0])
roi_counts_reshaped = np.reshape(roi_counts_sorted, (nr_rois, len(x_values), nr_avg))
print("reshaped data: (nr ROIs, nr x values, nr images): ", np.shape(roi_counts_reshaped))

# compute average by summing over repeated values 
counts_avg_perroi = np.mean(roi_counts_reshaped, axis=2)
counts_sem_perroi = np.std(roi_counts_reshaped, axis=2)/np.sqrt(nr_avg)

# %% plotting

# Plot number of counts as function of detuning for each ROI
fig2, axs = plt.subplots(figsize = (10,8), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))

# needs to be 1d to iterate. prep intial guess for fitting
axs = axs.ravel()
initial_guess = [8e3, 6e3, -2.1e6, 200e3] #  offset, amplitude, middle, width
popt_list = []

# X axis with more values for fit plot
x_axis_fit = np.linspace(x_values[0], x_values[-1], 100)

# plot data points
for roi_idx in range(nr_rois):
    axs[roi_idx].errorbar(x_values, counts_avg_perroi[roi_idx], 
        yerr=counts_sem_perroi[roi_idx], fmt='o', capsize=4, capthick=1)
    axs[roi_idx].set_title(f'ROI {roi_idx}')

    # fit datapoints
    popt, pcov = curve_fit(FittingFunctions.gaussian_function, x_values, counts_avg_perroi[roi_idx],
        p0=initial_guess)
        #, sigma=counts_sem_perroi[roi_idx], absolute_sigma=True)
    popt_list.append(popt)

    # plot fit result
    axs[roi_idx].plot(x_axis_fit, FittingFunctions.gaussian_function(x_axis_fit, *popt), color='red')

fig2.supxlabel('Detuning [Hz]')
fig2.supylabel('EMCCD Counts')

# %%
# Plot average over all ROIs as a function of detuning
avg_all_roi = np.mean(counts_avg_perroi, axis=0)
fig3, ax3 = plt.subplots()
ax3.errorbar(x_values, avg_all_roi, 
    yerr=np.std(counts_sem_perroi, axis=0), fmt='o', capsize=4, capthick=1, label='Counts')
ax3.set_xlabel('detuning [Hz]')
ax3.set_ylabel('EMCCD Counts')

if show_plots == True:
    plt.show() 

# calculate and print average and error of the peak location
popt_array = np.array(popt_list)
stark_shifts = popt_array[:, 2]
mean_stark_shift = np.mean(stark_shifts)
error_mean_stark_shift = np.std(stark_shifts)/np.sqrt(len(stark_shifts))
print("peak location", mean_stark_shift/1e3, "plusminus", error_mean_stark_shift/1e3) 

# %%
