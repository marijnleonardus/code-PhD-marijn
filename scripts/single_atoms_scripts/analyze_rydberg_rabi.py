# author marijn Venderbosch
# january 2026

"""first run histogram_and_threshold.py """
#%%

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
import pandas as pd

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from fitting_functions_class import FittingFunctions
from image_analysis_class import ImageStats
from single_atoms_class import SingleAtoms
from plot_utils import Plotting
from units import MHz

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

#%%

# variables
images_path = r"\\physstor\cqt-t\KAT1\Marijn\Scan123600"
file_name = os.path.join(images_path, "detection_threshold.npy")
binary_threshold = np.load(file_name)
roi_radius = 1
center_weight = 3

# load x values 
df = pd.read_csv(os.path.join(images_path, 'log.csv'))

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
fig1, axs = plt.subplots(figsize = (7, 6), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))