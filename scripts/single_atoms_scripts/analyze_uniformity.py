# author marijn Venderbosch
# April 2025

"""first run histogram_and_threshold.py """

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
import pandas as pd

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from fitting_functions_class import FittingFunctions
from image_analysis_class import ImageStats
from single_atoms_class import SingleAtoms
from plotting_class import Plotting
from units import MHz

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'T:\\KAT1\\Marijn\scan174612\\'
binary_threshold = np.load(images_path + 'detection_threshold.npy')
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
surv_prob = statistics_matrix[0]
sem_surv_prob = statistics_matrix[2]

# plot for each ROI the survival plobability as function of detuning and fit with Gaussian
nr_rois = np.shape(surv_matrix)[0]
fig1, axs = plt.subplots(figsize = (11, 9), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))

# needs to be 1d to iterate. prep intial guess for fitting
axs = axs.ravel()
initial_guess = [1, -0.5, 3.5e6, 200e3] #  offset, amplitude, middle, width
popt_list = []
pcov_list = []

# x axis with more values for fit plot
x_axis_fit = np.linspace(x_grid[0], x_grid[-1], 500)

for roi_idx in range(nr_rois):
    axs[roi_idx].errorbar(x_grid/MHz, surv_prob[roi_idx, :], sem_surv_prob[roi_idx, :],
        fmt='o', capsize=1, capthick=1)
    axs[roi_idx].set_title(f'ROI {roi_idx}')

    # fit datapoints and plot result
    popt, pcov = curve_fit(FittingFunctions.gaussian_function, x_grid, surv_prob[roi_idx, :], p0=initial_guess)
    popt_list.append(popt)
    pcov_list.append(pcov)
    axs[roi_idx].plot(x_axis_fit/MHz, FittingFunctions.gaussian_function(x_axis_fit, *popt), color='red')

fig1.supxlabel('Detuning [MHz]')
fig1.supylabel('Survival probabiility')

# calculate uniformity
detunings = np.array([arr[2] for arr in popt_list])
detunings_stddev = np.array([np.sqrt(pcov[2, 2]) for pcov in pcov_list])
print(detunings_stddev)
avg_detuning = np.mean(detunings)
sem_detuning = sem(detunings)
std_deviation_detuning = np.std(detunings)
uniformity = ImageStats.calculate_uniformity(detunings)

print("Fit position:  ", np.round(avg_detuning/MHz, 2), "p/m ", np.round(sem_detuning/MHz, 2), "MHz")
print("Standard deviation: ", 100*np.round(std_deviation_detuning/avg_detuning, 3), "%")
print("uniformity: ", np.round(uniformity, 3))

Plotting.savefig('output//','uniformity_plot.png')
