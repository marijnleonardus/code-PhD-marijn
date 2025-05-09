# author marijn Venderbosch
# April 2025

"""first run histogram_and_threshold.py """

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
from single_atoms_class import SingleAtoms
from plotting_class import Plotting
from units import kHz

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-04-11\\scan173216\\'
file_name_suffix = 'image'  # import files ending with image.tif
binary_threshold = np.load(images_path + 'detection_threshold.npy')

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_grid, surv_matrix = SingleAtomsStats.reshape_survival_matrix(df)
print("sorted data: nr ROIs, nr x_values: ", np.shape(surv_matrix))

# calculate mean and standard error per ROI and globally
statistics_matrix = SingleAtomsStats.calculate_survival_statistics(df) 
glob_surv = statistics_matrix[1]
glob_surv_sem = statistics_matrix[3]

fig1, ax1 = plt.subplots()
ax1.errorbar(x_grid/kHz, glob_surv, yerr=glob_surv_sem,
    fmt='o', color='blue', label='Array-averaged survival probability')

# fit surv. probability and plot
initial_guess = [0.95, 0.5, 10e3, 1e3] #  offset, amplitude, middle, width
popt, pcov = curve_fit(FittingFunctions.lorentzian, x_grid, glob_surv, p0=initial_guess)
x_axis_fit = np.linspace(x_grid[0], x_grid[-1], 500)
ax1.plot(x_axis_fit/kHz, FittingFunctions.lorentzian(x_axis_fit, *popt), color='red', label='Lorentzian fit')
ax1.set_xlabel('Modulation Freq.  [kHz]')
ax1.set_ylabel('Survival probabiility')
fig1.legend()

# obtain center and error in center
fit_center = popt[2]
perr = np.sqrt(np.diag(pcov))
fit_center_err = perr[2]
estimate_trap_freq = popt[2]/1.8 # factor 1.8 instead of 2 for anharmonicity of trap
print("fit result:" , np.round(fit_center/kHz,2), " pm ", np.round(fit_center_err/kHz,2),  "kHz")
print("estimate trap frequency: ", np.round(estimate_trap_freq/kHz, 2), "kHz")

Plotting().savefig('output//','surv_prob_fit_ax.png')