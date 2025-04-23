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

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-04-11\\scan154316\\'
kHz = 1e3
binary_threshold = 13200
roi_radius = 1
center_weight = 3

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_values, survival_probability = SingleAtomsStats.calculate_avg_survival(df)
print("sorted data: nr ROIs, nr x_values: ", np.shape(survival_probability))

# average over all ROIs
global_survival_probability = np.nanmean(survival_probability, axis=0)

fig1, ax1 = plt.subplots()
ax1.scatter(x_values/kHz, global_survival_probability)

# fit surv. probability and plot
initial_guess = [0.95, 0.5, 90e3, 1e3] #  offset, amplitude, middle, width
popt, pcov = curve_fit(FittingFunctions.lorentzian, x_values, global_survival_probability, p0=initial_guess)
x_axis_fit = np.linspace(x_values[0], x_values[-1], 500)
ax1.plot(x_axis_fit/kHz, FittingFunctions.lorentzian(x_axis_fit, *popt), color='red')

ax1.set_xlabel('Modulation Freq.  [kHz]')
ax1.set_ylabel('Survival probabiility')
plt.show()

fit_center = popt[2]
estimate_trap_freq = popt[2]/1.8 # factor 1.8 instead of 2 for anharmonicity of trap
print("fit result:" , np.round(fit_center/kHz,2), "kHz")
print("estimate trap frequency: ", np.round(estimate_trap_freq/kHz, 2), "kHz")

Plotting().savefig('output//','survival_probability_fit.png')