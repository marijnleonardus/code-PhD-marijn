# author: Marijn Venderbosch
# july 2024

import numpy as np
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
nr_bins_histogram = 20
images_path = 'Z:\\Strontium\\Images\\2025-03-06\\scan065218\\'
file_name_suffix = 'image'  # import files ending with image.tif
show_plots = True
log_threshold = 10 # laplacian of gaussian kernel sensitivity
weight_center_pixel = 5

# images without cropping ('raw' data)
image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)
images_list = [image_stack[i] for i in range(image_stack.shape[0])]

if np.shape(image_stack)[0] == 0:
    raise ValueError("No images loaded, check image path and file name suffix")
else:
    print("nr images, pixels, pixels", np.shape(image_stack))

# detect laplacian of gaussian spot locations from avg. over all images
z_project = np.mean(image_stack, axis=0)
#plt.imshow(z_project)
#plt.show()

spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=3, threshold=log_threshold)
y_coor = spots_LoG[:, 0] 
x_coor = spots_LoG[:, 1]
print(spots_LoG)
print("nr spots detected", np.shape(spots_LoG)[0])

# plot average image and mark detected maximum locations in red, check if LoG was correctly detected
fig1, ax1 = plt.subplots()
ax1.imshow(z_project)
ax1.scatter(x_coor, y_coor, marker='x', color='r')
fig1.show()
ax1.set_title('Average image and LoG detected spots')

ROI = RoiCounts(weight_center_pixel, rois_radius)
rois_matrix, roi_counts_matrix = ROI.compute_pixel_sum_counts(images_list, y_coor, x_coor)

# plot average pixel box for ROI 1 to check everythign went correctly
ROI.plot_average_of_roi(rois_matrix[0, :, :, :])

# Plot histograms for each ROI
nr_rois = np.shape(rois_matrix)[0]
fig2, axs = plt.subplots(ncols=int(np.sqrt(nr_rois)), nrows=int(np.sqrt(nr_rois)), sharex=True, sharey=True)
axs = axs.ravel()
for roi_idx in range(nr_rois):
    axs[roi_idx].hist(roi_counts_matrix[roi_idx, :], bins=nr_bins_histogram, edgecolor='black')
    axs[roi_idx].set_xlabel('Counts')
    axs[roi_idx].set_ylabel('Occurunces')

if show_plots == True:
    plt.show() 
