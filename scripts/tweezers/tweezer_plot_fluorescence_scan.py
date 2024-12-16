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
rois_radius = 2  # ROI size. Radius 1 means 3x3 array
images_path = 'Z:\\Strontium\\Images\\2024-12-10\\Scan135027\\'
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
else:
    print(np.shape(image_stack_raw))

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

# laod x_values
df = pd.read_csv(images_path + 'log.csv')
x_values = df.iloc[:, 0].to_numpy() # select right column, if mulitple averages used 

# Compute y-values per unique x-value. Average over identical x-values
x_values_unique = np.unique(x_values)
nr_avg = int(np.shape(x_values)[0]/np.shape(x_values_unique)[0])
nr_rois = np.shape(rois_matrix)[0]
roi_counts_reshaped = roi_counts_matrix.reshape(nr_rois, len(x_values_unique), nr_avg)
counts_avg_perroi = np.mean(roi_counts_reshaped, axis=2)
counts_std_perroi = np.std(roi_counts_reshaped, axis=2)

# Plot number of counts as function of detuning for each ROI
fig2, axs = plt.subplots(figsize = (10,8), sharex=True, sharey=True,
    ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)))

# needs to be 1d to iterate
axs = axs.ravel()

# rescale x axis
x_axis = x_values_unique/MHz

for roi_idx in range(nr_rois):
    axs[roi_idx].errorbar(x_axis, counts_avg_perroi[roi_idx], 
        yerr=counts_std_perroi[roi_idx], fmt='o', capsize=4, capthick=1, label='Counts')
    axs[roi_idx].set_title(f'ROI {roi_idx}')
    fig2.supxlabel('Detuning [MHz]')
    fig2.supylabel('EMCCD Counts')

# Plot average over all ROIs as a function of detuning
fig3, ax3 = plt.subplots()
ax3.errorbar(x_axis, np.mean(counts_avg_perroi, axis=0), 
    yerr=np.std(counts_std_perroi, axis=0), fmt='o', capsize=4, capthick=1, label='Counts')
ax3.set_xlabel('detuning [MHz]')
ax3.set_ylabel('EMCCD Counts')

if show_plots == True:
    plt.show() 

