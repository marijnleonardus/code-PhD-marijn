import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'T:\\KAT1\\Marijn\scan174612\\'
MHz = 1e6
binary_threshold = 15200
roi_radius = 1
center_weight = 3

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_values, survival_probability = SingleAtomsStats.calculate_avg_survival(df)
print("sorted data: nr ROIs, nr x_values: ", np.shape(survival_probability))

nr_rois = survival_probability.shape[0]

# plot for each ROI the survival plobability as function of detuning and fit with Gaussian
fig1, axs = plt.subplots(figsize = (10, 8), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))

# needs to be 1d to iterate. prep intial guess for fitting
axs = axs.ravel()
initial_guess = [1, -0.5, 3.5e6, 200e3] #  offset, amplitude, middle, width
popt_list = []

# x axis with more values for fit plot
x_axis_fit = np.linspace(x_values[0], x_values[-1], 500)

for roi_idx in range(nr_rois):
    axs[roi_idx].scatter(x_values/MHz, survival_probability[roi_idx, :])
    axs[roi_idx].set_title(f'ROI {roi_idx}')

    # fit datapoints and plot result
    popt, pcov = curve_fit(FittingFunctions.gaussian_function, x_values, survival_probability[roi_idx, :], p0=initial_guess)
    popt_list.append(popt)
    axs[roi_idx].plot(x_axis_fit/MHz, FittingFunctions.gaussian_function(x_axis_fit, *popt), color='red')

fig1.supxlabel('Detuning [MHz]')
fig1.supylabel('Survival probabiility')
plt.show()

detunings = np.array([arr[2] for arr in popt_list])
avg_detuning = np.mean(detunings)
std_deviation = np.std(detunings)
uniformity = ImageStats.calculate_uniformity(detunings)
print("Average middle fit fit: ", np.round(avg_detuning/MHz, 2), "MHz")
print("Standard deviation: ", np.round(std_deviation/MHz, 2), "MHz")
print("uniformity: ", np.round(uniformity, 3))
