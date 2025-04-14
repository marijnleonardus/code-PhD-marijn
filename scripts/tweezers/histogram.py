# author: Marijn Venderbosch
# April 2025

import numpy as np
import os
import matplotlib.pyplot as plt

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'T:\\KAT1\\Marijn\scan174612\\selection'
nr_bins_histogram = 20

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
    axs[roi_idx].hist(roi_counts_matrix[roi_idx, :], bins=nr_bins_histogram, edgecolor='black')
    axs[roi_idx].set_xlabel('Counts')
    axs[roi_idx].set_ylabel('Occurunces')

# --- Additional part: Plot the Average Histogram of all ROIs ---
# Compute the average count per ROI (i.e., average across images)
counts_matrix = roi_counts_matrix.ravel()
print(counts_matrix[0])

# Plotting the histogram of these average counts
fig2, ax2 = plt.subplots()
ax2.hist(counts_matrix, bins=5*nr_bins_histogram, edgecolor='black')
ax2.set_xlabel('Counts')
ax2.set_ylabel('Occurunces')
plt.grid(True)
plt.show()
