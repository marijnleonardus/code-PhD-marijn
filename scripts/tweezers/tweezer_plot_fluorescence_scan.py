# author: Marijn Venderbosch
# December 2024

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from scipy.optimize import curve_fit

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from camera_image_class import CameraImage
from image_analysis_class import ManipulateImage, RoiCounts
from fitting_functions_class import FittingFunctions

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
rois_radius = 2  # ROI size. Radius 1 means 3x3 array
images_path = 'Z:\\Strontium\\Images\\2024-12-10\\scan140135\\'
file_name_suffix = 'image'  # import files ending with image.tif
show_plots = True
log_threshold = 80 # laplacian of gaussian kernel sensitivity
weight_center_pixel = 1 # if weighted pixel box is to be used
crop_images = True
crop_radius = 17
crop_y_center = 35
crop_x_center = 40

MHz = 1e6

# images without cropping ('raw' data)
image_stack_raw = CameraImage().import_image_sequence(images_path, file_name_suffix)

if np.shape(image_stack_raw)[0] == 0:
    raise ValueError("No images loaded, check image path and file name suffix")

# crop images in the 3d np arraqy
if crop_images:
    nr_images = np.shape(image_stack_raw)[0]
    cropped_pixel_dim = 2*crop_radius + 1
    image_stack = np.zeros((nr_images, cropped_pixel_dim, cropped_pixel_dim), dtype=int)
    for img in range(nr_images):
        image_stack[img] = ManipulateImage().crop_array_center(image_stack_raw[img, :, :],
            crop_y_center, crop_x_center, crop_radius)
else:
    image_stack = image_stack_raw

# detect laplacian of gaussian spot locations from avg. over all images
z_project = np.mean(image_stack, axis=0)
spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=3, threshold=log_threshold)
y_coor = spots_LoG[:, 0] 
x_coor = spots_LoG[:, 1]

# plot average image and mark detected maximum locations in red, check if LoG was correctly detected
fig1, ax1 = plt.subplots()
ax1.imshow(z_project)
ax1.scatter(x_coor, y_coor, marker='x', color='r')
plt.show()

# return list form for function ROI pixel sum calculator
images_list = [image_stack[i] for i in range(image_stack.shape[0])]

# obtain counts over all ROIs
ROI = RoiCounts(weight_center_pixel, rois_radius)
rois_matrix, roi_counts_matrix = ROI.compute_pixel_sum_counts(images_list, y_coor, x_coor)

# plot average pixel box for ROI 1 to check everything went correctly
ROI.plot_average_of_roi(rois_matrix[0, :, :, :])

# laod x_values. If multiple averages used x values contains duplicates
df = pd.read_csv(images_path + 'log.csv')
x_values_duplicates = df.iloc[:, 0].to_numpy() 

# remove x duplicates. Compute y-values per unique x-value. Average over identical x-values
x_values = np.unique(x_values_duplicates)
nr_avg = int(np.shape(x_values_duplicates)[0]/np.shape(x_values)[0])
nr_rois = np.shape(rois_matrix)[0]
roi_counts_reshaped = roi_counts_matrix.reshape(nr_rois, len(x_values), nr_avg)

# calculate average and standard error mean (SEM)
counts_avg_perroi = np.mean(roi_counts_reshaped, axis=2)
counts_sem_perroi = np.std(roi_counts_reshaped, axis=2)/np.sqrt(nr_avg)

# Plot number of counts as function of detuning for each ROI
fig2, axs = plt.subplots(figsize = (10,8), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))

# needs to be 1d to iterate
axs = axs.ravel()

# fitting guess for gaussian fits. And empty array for saving fitting results. X axis with more values for fit plot
initial_guess = [14000, 5000, -2e6, 0.5e6]
popt_list = []
x_axis_fit = np.linspace(x_values[0], x_values[-1], 100)

# plot data points
for roi_idx in range(nr_rois):
    axs[roi_idx].errorbar(x_values, counts_avg_perroi[roi_idx], 
        yerr=counts_sem_perroi[roi_idx], fmt='o', capsize=4, capthick=1)
    axs[roi_idx].set_title(f'ROI {roi_idx}')

    # fit datapoints
    popt, pcov = curve_fit(FittingFunctions.gaussian_function, x_values, counts_avg_perroi[roi_idx],
        p0=initial_guess, sigma=counts_sem_perroi[roi_idx], absolute_sigma=True)
    popt_list.append(popt)

    # plot fit result
    axs[roi_idx].plot(x_axis_fit, FittingFunctions.gaussian_function(x_axis_fit, *popt), color='red')
fig2.supxlabel('Detuning [Hz]')
fig2.supylabel('EMCCD Counts')

# Plot average over all ROIs as a function of detuning
avg_all_roi = np.mean(counts_avg_perroi, axis=0)
fig3, ax3 = plt.subplots()
ax3.errorbar(x_values, avg_all_roi, 
    yerr=np.std(counts_sem_perroi, axis=0), fmt='o', capsize=4, capthick=1, label='Counts')
ax3.set_xlabel('detuning [Hz]')
ax3.set_ylabel('EMCCD Counts')

if show_plots == True:
    plt.show() 

popt_array = np.array(popt_list)
stark_shifts = popt_array[:, 2]
mean_stark_shift = np.mean(stark_shifts)
error_mean_stark_shift = np.std(stark_shifts)/np.sqrt(len(stark_shifts))
print("peak location", mean_stark_shift/1e3, "plusminus", error_mean_stark_shift/1e3)
