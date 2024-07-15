
# %% imports

import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import blob_log

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from image_analysis import ManipulateImage, LoadImageData

# %% variables

rois_radius = 0
nr_bins_histogram = 50
images_path = 'Z://Strontium//Images//2024-07-11//scan335929//'
file_name_suffix = '0000image'
crop_pixels_x = 20
crop_pixels_y = 20 

# %% import images and crop images

# images without cropping ('raw' data)
image_stack_raw = LoadImageData().import_image_sequence(images_path, file_name_suffix)

# crop images to remove spurious noise peaks at the edges
image_stack = []
for img in image_stack_raw:
    cropped_array = ManipulateImage().crop_array_edge(img, crop_pixels_x, crop_pixels_y)
    image_stack.append(cropped_array)
image_stack = np.array(image_stack)

# print("nr_images, nr_rows, nr_columns = ", image_stack.shape)

# %% compute average image and peak locations

# compute cropped average image and plot
z_project = np.mean(image_stack, axis=0)

# compute laplacian of gaussian spot locations
spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=10, threshold=50)

# return rows, columns of detected spots 
y_coor = spots_LoG[:, 0]
x_coor = spots_LoG[:, 1]

# store in a 2d array for convenient passing down to other functions
rois_array = np.column_stack((x_coor, y_coor))

# print("ROIs: ", rois_array)

# plot average image and mark detected maximum locations in red
fig0, ax0 = plt.subplots()
ax0.imshow(z_project)
ax0.scatter(x_coor, y_coor, marker='x', color='r')
fig0.show()

# %% read counts within each ROI

def read_counts_within_rois(rois_array, stack_of_images):
    nr_rois = int(rois_array.shape[0])
    nr_images = int(stack_of_images.shape[0])
    
    # Initialize a matrix to store counts with shape (num_rois, num_images)
    counts_matrix = np.zeros((nr_rois, nr_images), dtype=int)
    
    # Iterate through each image and ROI
    for image_i in range(nr_images):
        image = stack_of_images[image_i]
        
        for roi_i in range(nr_rois):
            col, row = rois_array[roi_i]
            row = int(row)
            col = int(col)
            
            if rois_radius > 0:
                # select an ROI area of multiple pixels 
                roi_region = ManipulateImage().crop_array_center(image, col, row, rois_radius)
                counts = np.sum(roi_region)
            else:
                # only select the one pixel corresponding to the maximum
                counts = image[row, col]
            
            counts_matrix[roi_i, image_i] = counts
    return counts_matrix

counts_matrix = read_counts_within_rois(rois_array, image_stack)

counts_matrix = np.array(counts_matrix)
print("nr ROIs, nr counts = ", counts_matrix.shape)

# %% plot histograms

# Create a figure and axes for the subplots
num_rois = len(rois_array)
array_dim = int(np.sqrt(num_rois))
fig, axes = plt.subplots(nrows=array_dim, ncols=array_dim, figsize=(12, 12), constrained_layout=True)
ax = axes.flatten()

# Plot histograms for each ROI
for roi in range(num_rois):
    ax[roi].hist(counts_matrix[roi], bins=nr_bins_histogram, edgecolor='black')
    ax[roi].set_title(f'Histogram of Counts for ROI {roi+1}')
    ax[roi].set_xlabel('Counts')
    ax[roi].set_ylabel('Frequency')

# %%
plt.show()