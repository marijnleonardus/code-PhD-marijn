# author marijn Venderbosch
# April 2025

"""first run histogram_and_threshold.py """

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.constants import pi

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from single_atoms_class import SingleAtoms
from plotting_class import Plotting
from fitting_functions_class import FittingFunctions

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-04-28\\scan203308\\'
us = 1e-6 # microseconds
kHz = 1e3 # kilohertz
roi_radius = 1
center_weight = 3
truncate_x_values = 13

# load binary threshold from histrogram script
binary_threshold = np.load(images_path + 'detection_threshold.npy')

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_values, surv_prob,_ , error_global_surv_prob = SingleAtomsStats.calculate_avg_sem_survival(df)
global_surv_prob = np.nanmean(surv_prob, axis=0)
print("sorted data: nr ROIs, nr x_values: ", np.shape(surv_prob))

# truncate data
if truncate_x_values > 0:
    x_values = x_values[:truncate_x_values]
    global_surv_prob = global_surv_prob[:truncate_x_values]
    error_global_surv_prob = error_global_surv_prob[:truncate_x_values]

# fit data with damped sin
initial_guess = [0.15, 0.01/us, 60*kHz, 3*pi/2, 0.35]
bounds = (0, [0.2, 1/us, 100*kHz, 2*pi, 0.5])
popt, pcov = curve_fit(FittingFunctions.damped_sin_wave, x_values, global_surv_prob, 
    p0=initial_guess, bounds=bounds)    
print(popt)
x_values_fit = np.linspace(0, np.max(x_values), 100)

fig1, ax1 = plt.subplots()
ax1.errorbar(x_values/us, global_surv_prob, yerr=error_global_surv_prob,
    fmt='o', color='blue', label='survival probability')
ax1.set_xlabel(r'Release time [$\mu$s]')
ax1.set_ylabel('Survival probabiility')

ax1.plot(x_values_fit/us, FittingFunctions.damped_sin_wave(x_values_fit, *popt), color='red')

Plotting().savefig('output//','release_recapture_fit.png') 
