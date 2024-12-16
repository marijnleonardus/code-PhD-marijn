# author: Marijn Venderbosch
# December 2024

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.feature import blob_log

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from camera_image_class import CameraImage
from image_analysis_class import ManipulateImage, RoiCounts

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
rois_radius = 1  # ROI size. Radius 1 means 3x3 array
images_path = 'Z:\\Strontium\\Images\\2024-12-10\\scan154351\\'
file_name_suffix = 'image'  # import files ending with image.tif
show_plots = True
log_threshold = 40 # laplacian of gaussian kernel sensitivity
weight_center_pixel = 3
column_xvalues = 0 # column with independent variables

MHz = 1e6

# images without cropping ('raw' data)
image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)
images_list = [image_stack[i] for i in range(image_stack.shape[0])]

# detect laplacian of gaussian spot locations from avg. over all images
z_project = np.mean(image_stack, axis=0)
spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=3, threshold=log_threshold)
y_coor = spots_LoG[:, 0] 
x_coor = spots_LoG[:, 1]

# plot average image and mark detected maximum locations in red, check if LoG was correctly detected
fig1, ax1 = plt.subplots()
ax1.imshow(z_project)
ax1.scatter(x_coor, y_coor, marker='x', color='r')

# obtain counts over all ROIs
ROI = RoiCounts(weight_center_pixel, rois_radius)
rois_matrix, roi_counts_matrix = ROI.compute_pixel_sum_counts(images_list, y_coor, x_coor)

# plot average pixel box for ROI 1 to check everything went correctly
ROI.plot_average_of_roi(rois_matrix[0, :, :, :])

# laod detunings
df = pd.read_csv(images_path + 'log.csv')
detunings = df.iloc[:, column_xvalues].to_numpy() # select right column, if mulitple averages used 

# Compute y-values per unique x-value. Average over identical x-values
detunings_unique = np.unique(detunings)
nr_avg_datapoint = int(np.shape(detunings)[0]/np.shape(detunings_unique)[0])
nr_rois = np.shape(rois_matrix)[0]
roi_counts_reshaped = roi_counts_matrix.reshape(nr_rois, len(detunings_unique), nr_avg_datapoint)

roi_counts_avg = np.mean(roi_counts_reshaped, axis=2)
roi_counts_std = np.std(roi_counts_reshaped, axis=2)

# Plot number of counts as function of detuning for each ROI
fig2, axs = plt.subplots(figsize = (10,8), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))
axs = axs.ravel()
for roi_idx in range(nr_rois):
    axs[roi_idx].errorbar(detunings_unique/MHz, roi_counts_avg[roi_idx], 
        yerr=roi_counts_std[roi_idx], fmt='o', capsize=4, capthick=1, label='Counts')
    axs[roi_idx].set_title(f'ROI {roi_idx}')
    fig2.supxlabel('Detuning [MHz]')
    fig2.supylabel('EMCCD Counts')

if show_plots == True:
    plt.show() 
