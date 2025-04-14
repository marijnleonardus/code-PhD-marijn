# author: Marijn Venderbosch
# April 2025

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import pi

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from fitting_functions_class import FittingFunctions

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'T:\\KAT1\\Marijn\scan174612\\selection'
nr_bins_hist_roi = 15
nr_bins_hist_avg = 50

# load ROI counts from npy
# (nr ROIs, nr images)
roi_counts_matrix = np.load(os.path.join(images_path, 'roi_counts_matrix.npy'))
print("nr ROIs, nr images")
print(np.shape(roi_counts_matrix))

# Plot histograms for each ROI
nr_rois = np.shape(roi_counts_matrix)[0]
fig1, ax1 = plt.subplots(ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)),
    sharex=True, sharey=True)

axs = ax1.ravel()
for roi_idx in range(nr_rois):
    axs[roi_idx].hist(roi_counts_matrix[roi_idx, :], bins=nr_bins_hist_roi, edgecolor='black')
    axs[roi_idx].set_xlabel('Counts')
    axs[roi_idx].set_ylabel('Occurunces')

# average histrogram over all ROIS from summing the ROIs
counts_matrix = roi_counts_matrix.ravel()

# fit histogram with gaussian function
hist_vals, bin_edges = np.histogram(counts_matrix, bins=nr_bins_hist_avg)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2 

initial_guess = [max(hist_vals), np.mean(counts_matrix)*0.8, np.std(counts_matrix)*0.5,
    max(hist_vals)/2, np.mean(counts_matrix)*1.2, np.std(counts_matrix)*0.5]

# Fit the model to the histogram data
popt, _ = curve_fit(FittingFunctions.double_gaussian, bin_centers, hist_vals, p0=initial_guess)

# x values for plotting the fitted curve
x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
y_fit = FittingFunctions.double_gaussian(x_fit, *popt)

# plot avg histogram
fig2, ax2 = plt.subplots()
ax2.hist(counts_matrix, bins=nr_bins_hist_avg, edgecolor='black')
ax2.set_xlabel('Counts')
ax2.set_ylabel('Occurunces')
plt.grid(True)
ax2.plot(x_fit, y_fit, 'r-', label='Double Gaussian fit')

# compute peaks area ratio
area_0 = np.sqrt(2*pi)*popt[0]*popt[2]
area_1 = np.sqrt(2*pi)*popt[3]*popt[5]
area_1_ratio = area_1/(area_0 + area_1)
print("area_1_ratio", area_1_ratio)

plt.show()
