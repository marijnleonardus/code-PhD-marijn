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
from image_analysis_class import ManipulateImage, Histograms

# variables
rois_radius = 1  # ROI size. Radius 1 means 3x3 array
nr_bins_histogram = 50
images_path = 'Z://Strontium//Images//2024-10-10//scan114820//'
file_name_suffix = 'image'  # import files ending with image.tif
crop_pixels_x = 5  # amount of columns to remove left and right
crop_pixels_y = 5  # columns to remove top and bottom
show_plots = True
weighted_sum = True
weight_center_pixel = 3 # when computing weighted sum, relative contribution center pixel

# images without cropping ('raw' data)
image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)

# compute cropped average image and plot
z_project = np.mean(image_stack, axis=0)

# compute laplacian of gaussian spot locations
spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=10, threshold=50)

# return rows, columns of detected spots 
y_coor = spots_LoG[:, 0] + 1 # no idea why +1, but i don't do it the cropped ROIs look off by 1 pixel
x_coor = spots_LoG[:, 1] 

# store in a 2d array for convenient passing down to other functions
rois_array = np.column_stack((x_coor, y_coor))

# plot average image and mark detected maximum locations in red
fig0, ax0 = plt.subplots()
ax0.imshow(z_project)
ax0.scatter(x_coor, y_coor, marker='x', color='r')
fig0.show()

def read_counts_within_rois(rois, stack_of_images):
    """given a set (stack) of images and ROIs set, 
    computes the number of counts within each ROI within each image

    Args:
        rois (2d np array): set of the maxima locations
        stack_of_images (3d np array): set of 2d images (np arrays)

    Returns:
        rois_list (list): list of all ROI regions for all images
        counts_matrix (np array): matrix of counts within each ROI for each image
    """
    num_regions = len(rois)
    num_images = len(stack_of_images)
    
    counts_matrix = np.zeros((num_regions, num_images), dtype=int)
    rois_list = []

    for image_index, image in enumerate(stack_of_images):
        for region_index, (col, row) in enumerate(rois):
            if rois_radius > 0:
                roi_region = ManipulateImage().crop_array_center(
                    image, col, row, rois_radius)
                rois_list.append(roi_region)
                if weighted_sum == True:
                    counts = Histograms().weighted_count_roi(weight_center_pixel, roi_region)
                else:
                    counts = np.sum(roi_region)
            else:
                counts = image[int(row), int(col)]
            counts_matrix[region_index, image_index] = counts
    return (rois_list, counts_matrix)


regions_list, roi_counts_matrix = read_counts_within_rois(rois_array, image_stack)

# all regions of interest for all ROIs and all images
regions_array = np.array(regions_list)
print(regions_array)

# compute average ROI for all ROIs of all images and ROIs
avg_roi = np.mean(regions_array, axis=0)
fig1, ax1 = plt.subplots()
ax1.imshow(avg_roi)

# plot histograms
num_rois = len(rois_array)
array_dim = int(np.sqrt(num_rois))

fig2, axes = plt.subplots(nrows=array_dim, ncols=array_dim, figsize=(12, 12), 
    sharex=True, sharey=True, constrained_layout=True)
ax = axes.flatten() # needs to be a 1d array for the for loop

# Plot histograms for each ROI
for roi in range(num_rois):
    ax[roi].hist(roi_counts_matrix[roi], bins=nr_bins_histogram, edgecolor='black')
    ax[roi].set_title(f'Histogram of Counts for ROI {roi+1}')
    ax[roi].set_xlabel('Counts')
    ax[roi].set_ylabel('Frequency')

if show_plots == True:
    plt.show()
