# author: Marijn Venderbosch
# April 2025

""""script that calculates the detection threshold from a histogram of the counts in the ROIs

saves to the files 
- detection_threshold.npy
- roi_counts_matrix.npy

that are used by other analysis scripts
"""

# %%

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
from single_atoms_class import ROIs
from camera_image_class import EMCCD
from plotting_class import Plotting

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-04-01\\scan104728\\'
file_name_suffix = 'image'  # import files ending with image.tif
nr_bins_hist_roi = 30
nr_bins_hist_avg = 50
roi_radius = 2

# %% 

# Calculate counts in each ROI and save to npy array for other functions to use
ROIsObject = ROIs(roi_radius)
roi_counts_matrix = ROIsObject.calculate_roi_counts(images_path, file_name_suffix)
print("raw data: (nr ROIs, nr images): ", np.shape(roi_counts_matrix))
np.save(images_path + 'roi_counts_matrix.npy', roi_counts_matrix)

# %% 

# Plot histograms for each ROI
nr_rois = np.shape(roi_counts_matrix)[0]
fig1, ax1 = plt.subplots(ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)),
    sharex=True, sharey=True)
axs = ax1.ravel()
for roi_idx in range(nr_rois):
    axs[roi_idx].hist(roi_counts_matrix[roi_idx, :], bins=nr_bins_hist_roi, edgecolor='black')
fig1.supxlabel('EMCCD Counts')
fig1.supylabel('Occurences')

# change to 1D matrix for histogram computation averaged over all ROIs
counts = roi_counts_matrix.ravel()

# fit histogram with gaussian function
hist_vals, bin_edges = np.histogram(counts, bins=nr_bins_hist_avg)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2 
initial_guess = [max(hist_vals), np.mean(counts)*0.8, np.std(counts)*0.5, max(hist_vals)/4, np.mean(counts)*1.2, np.std(counts)]
fit_boundaries = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
popt, _ = curve_fit(FittingFunctions.double_gaussian, bin_centers, hist_vals, p0=initial_guess, bounds=fit_boundaries)
x_fit_counts = np.linspace(bin_centers[0], bin_centers[-1], 1000)
y_fit_counts = FittingFunctions.double_gaussian(x_fit_counts, *popt)

# calculate detection threshold
detection_treshold_counts = ROIs.calculate_histogram_detection_threshold(popt)
print(detection_treshold_counts)
np.save(images_path + 'detection_threshold.npy', detection_treshold_counts)

# calculate area 1 peak
filling_fraction = np.sum(counts > detection_treshold_counts)/len(counts)
print(f"Filling fraction: {filling_fraction:.3f}")

# plot avg histogram using EMCCD counts as x axis
fig2, ax2 = plt.subplots()
ax2.set_xlabel('EMCCD Counts')
ax2.set_ylabel('Occurences')
ax2.hist(counts, bins=nr_bins_hist_avg, edgecolor='black')

plt.grid(True)
ax2.plot(x_fit_counts, y_fit_counts, 'r-', label='Double Gaussian fit')
ax2.axvline(detection_treshold_counts, color='grey', linestyle='--', label='Detection threshold')
fig2.legend()

# same histogram but rescaled in terms of photon number 
center_0_peak = popt[1]
photons_matrix = EMCCD.counts_to_photons(counts, center_0_peak)
detection_threshold_photons = EMCCD.counts_to_photons(detection_treshold_counts, center_0_peak)
x_fit_photons = EMCCD.counts_to_photons(x_fit_counts, center_0_peak)

# plot scaled histogram
fig3, ax3 = plt.subplots()
ax3.set_xlabel('Number of photons')
ax3.set_ylabel('Occurences')
ax3.hist(photons_matrix, bins=nr_bins_hist_avg, edgecolor='black') # don't know why i need minus here
ax3.grid(True)
ax3.plot(x_fit_photons, y_fit_counts, 'r-', label='Double Gaussian fit')
ax3.axvline(detection_threshold_photons, color='grey', linestyle='--', label='Detection threshold')
fig3.legend()

# %%

Plotting.savefig("output//", 'roi_histogram.png')
# %%
