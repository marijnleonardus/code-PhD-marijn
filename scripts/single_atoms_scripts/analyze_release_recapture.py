# author marijn Venderbosch
# April 2025

"""first run histogram_and_threshold.py """

import numpy as np
import os
import matplotlib.pyplot as plt
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
images_path = 'Z:\\Strontium\\Images\\2025-04-01\\scan104728\\'
us = 1e-6 # microseconds
binary_threshold = 13680
roi_radius = 1
center_weight = 3

# load x values 
df = pd.read_csv(images_path + 'log.csv')

# calculate survival probability from x values, threshold and images list
SingleAtomsStats = SingleAtoms(binary_threshold, images_path)
x_values, survival_probability, err_survival_probability = SingleAtomsStats.calculate_avg_sem_survival(df)
print("sorted data: nr ROIs, nr x_values: ", np.shape(survival_probability))

# average over all ROIs
global_survival_probability = np.nanmean(survival_probability, axis=0)
nr_rois = np.shape(survival_probability)[0]
global_err_survival_probability = np.nanmean(err_survival_probability, axis=0)/np.sqrt(nr_rois)

fig1, ax1 = plt.subplots()
ax1.errorbar(x_values/us, global_survival_probability, yerr=global_err_survival_probability, fmt='o', color='blue', label='survival probability')
ax1.set_xlabel(r'Release time [$\mu$s]')
ax1.set_ylabel('Survival probabiility')

Plotting().savefig('output//','release_recapture_fit.png') 
