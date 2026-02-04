# author: Marijn Venderbosch
# April 2025

""""script that
* calculates ROI counts in a dataset
* calculates the histogram, and corresopnding optimal detection threshold
* saves these processed data to output folder to be used by follow up scripts
"""


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import shutil

# append modules dir
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from fitting_functions_class import FittingFunctions
from single_atoms_class import ROIs, BinaryThresholding
from camera_image_class import EMCCD
from plot_utils import Plotting

os.system('cls' if os.name == 'nt' else 'clear')

# variables
# images_path = 'Z:\\Strontium\\Images\\2026-01-23\\scan172311\\'

images_path = 'Z:\\Strontium\\Images\\2026-01-27\\'
rid = 'scan193823'
# single atom rabi oscillations
#images_path = 'Z:\\Strontium\\Images\\2026-01-28\\'
#rid = 'scan193223'

# random 200 point dataset for debugging
#images_path = 'Z:\\Strontium\\Images\\2026-01-29\\'
#rid = 'scan164420'

# clock rabi
images_path = 'Z:\\Strontium\\Images\\2026-02-04\\'
rid = 'scan001915'

# RoI geometry
nr_rows = 3
nr_cols = 3

file_name_suffix = 'image'  # import files ending with image.tiff
roi_index_tolerance = 4
nr_bins_hist_roi = 12
nr_bins_hist_avg = 50
roi_radius = 1
log_thresh = 10
plot_only_initial = True # of each set of 2 images (inital, survival) throw away survival
show_photon_histogram = False # show histogram of counts per photon

# Calculate counts in each ROI using weighted pixel boxes 
ROIsObject = ROIs(roi_radius, log_thresh)
import_path = images_path + rid + '\\'
spots, roi_counts_matrix = ROIsObject.calculate_roi_counts(import_path, file_name_suffix, use_weighted_count=True, roi_index_tolerance=roi_index_tolerance)
print("raw data: (nr ROIs, nr shots): ", np.shape(roi_counts_matrix))

# save to output folder to be used for other scripts
output_path = Path(f'output/processed_data/roi_counts/{rid}')
output_path.mkdir(parents=True, exist_ok=True)
np.save(output_path / 'roi_counts_matrix.npy', roi_counts_matrix)

# compute histograms for each ROI
if plot_only_initial:
    roi_counts_matrix = roi_counts_matrix[:, ::2]

nr_rois = np.shape(roi_counts_matrix)[0]
fig1, ax1 = plt.subplots(nrows=nr_rows, ncols=nr_cols, 
    figsize=(2*nr_cols, 2*nr_rows), # Dynamic scaling
    sharex=True, sharey=True)

# Flatten axs in case of a 1D grid or a single ROI
axs = np.atleast_1d(ax1).ravel()

# plot each ROI
for roi_idx in range(nr_rois):
    axs[roi_idx].hist(roi_counts_matrix[roi_idx, :], bins=nr_bins_hist_roi, edgecolor='black')
fig1.supxlabel('EMCCD Counts')
fig1.supylabel('Occurences')

# change to 1D matrix for histogram computation averaged over all ROIs
counts = roi_counts_matrix.ravel()

# make histogram
hist_vals, bin_edges = np.histogram(counts, bins=nr_bins_hist_avg)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2 

# fit double gaussian to histogram
#print('mean, stddev in counts are ', np.round(np.mean(counts)), np.round(np.std(counts)))
init_guess = [max(hist_vals), np.mean(counts)*0.9, np.std(counts)*0.5,
    max(hist_vals)/4, np.mean(counts)*1.1, np.std(counts)]
fit_limits = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
popt, pcov = curve_fit(FittingFunctions.double_gaussian, bin_centers, hist_vals, p0=init_guess, bounds=fit_limits)

# produce data with finer grid for ploting double gaussian function
x_fit_counts = np.linspace(bin_centers[0], bin_centers[-1], 200)
y_fit_counts = FittingFunctions.double_gaussian(x_fit_counts, *popt)

# calculate detection threshold
DoubleGauss = BinaryThresholding(popt)

# calculate filling fraction from the fit itself, which we need to find the detection threshold (confusing)
# later we calculate the actual filling fraction from the EMMCCD counts, but we can only do this when we know the filling fraction
# Area ~ Amplitude * Sigma (the sqrt(2pi) cancels out in the fraction)
area_background = popt[0]*popt[2]
area_signal = popt[3]*popt[5]
filling_fraction_from_fit = area_signal/(area_background + area_signal)
print(f"Filling fraction derived from fit: {filling_fraction_from_fit:.3f}")
      
detection_treshold_counts = DoubleGauss.calculate_histogram_detection_threshold(filling_fraction=filling_fraction_from_fit)
print('detection threshold: ', detection_treshold_counts)
np.save(output_path / 'detection_threshold.npy', detection_treshold_counts)

# calculate area 1 peak
filling_fraction = np.sum(counts > detection_treshold_counts)/len(counts)
print(f"Filling fraction obtained from EMCCD counts: {filling_fraction:.3f}")
fidelity = DoubleGauss.calculate_imaging_fidelity(filling_fraction)
print(f"Imaging fidelity: {fidelity:.5f}")

## plotting
# plot avg histogram using EMCCD counts as x axis
fig2, ax2 = plt.subplots()
ax2.set_xlabel('EMCCD Counts')
ax2.set_ylabel('Occurences')
ax2.hist(counts, bins=nr_bins_hist_avg, edgecolor='black')

plt.grid(True)
ax2.plot(x_fit_counts, y_fit_counts, 'r-', label='Double Gaussian fit')
ax2.axvline(detection_treshold_counts, color='grey', linestyle='--', label='Detection threshold')
ax2.legend()

if show_photon_histogram:
    # same histogram but rescaled in terms of photon number 
    # convert photon number from Ixon conversion formula using background count nr
    backgr_counts = popt[1]
    iXon888 = EMCCD()
    photons_matrix = iXon888.counts_to_photons(counts, backgr_counts)
    detection_threshold_photons = iXon888.counts_to_photons(detection_treshold_counts, backgr_counts)
    x_fit_photons = iXon888.counts_to_photons(x_fit_counts, backgr_counts)

    # we need to apply a scaling factor to correct for the weighted pixel count
    # this will disturb the absolute photon number. 
    # Rescale by computing the non-weighted pixel count
    roi_counts_matrix_non_weighted = ROIsObject.calculate_roi_counts(import_path, file_name_suffix, use_weighted_count=False)
    photons_matrix_non_weighted = iXon888.counts_to_photons(roi_counts_matrix_non_weighted.ravel(), backgr_counts)
    rescale_factor = photons_matrix_non_weighted.mean()/photons_matrix.mean()

    fig_width = 3.375*.5 - 0.02  # inches, matches one column
    fig_height = fig_width*0.61
    fig3, ax3 = plt.subplots(figsize = (fig_width, fig_height))
    ax3.set_xlabel('Number of photons')
    ax3.set_ylabel('Probability')
    ax3.grid()
    ax3.hist(photons_matrix*rescale_factor, bins=nr_bins_hist_avg, edgecolor='black', density=True, label='Counts') 
    ax3.axvline(detection_threshold_photons*rescale_factor, color='grey', linestyle='--', label='Detection threshold')

    print(detection_threshold_photons*rescale_factor)

    Plotting = Plotting('output')
    Plotting.savefig('roi_histogram.pdf')

# save files to be used by other scripts
np.savetxt(output_path / "popt.csv", popt, delimiter = ',')
np.savetxt(output_path / "pcov.csv", pcov, delimiter = ',')
np.savetxt(output_path / "roi_geometry.csv", [nr_rows, nr_cols], fmt="%d", delimiter=",")
np.save(output_path / "filling_fraction.npy", filling_fraction)

# also copy the `log.csv` file, that is needed by other scripts, to private folder
source_log = Path(import_path) / 'log.csv'
destination_log = output_path / 'log.csv'
if source_log.exists():
    shutil.copy2(source_log, destination_log)
    print(f"Successfully copied log.csv to {output_path}")
else:
    print(f"Warning: log.csv not found in {import_path}")

plt.show()
