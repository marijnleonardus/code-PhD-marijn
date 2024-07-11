
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

from image_analysis import ManipulateImage

# %% variables

rois_radius = 2
nr_bins_histogram = 20
images_path = 'Z://Strontium//Images//2024-07-11//scan335929//'
crop_pixels = 15 

# %% import images

# i want the first image of each sequence. they end with 0000image
images_filenames = glob.glob(os.path.join(images_path, '*0000image.tif'))

# create a 3d array of all 2d images stacked 
image_stack = []
for file_name in images_filenames:
    img = np.array(Image.open(file_name))
    cropped_array = ManipulateImage().crop_array(img, crop_pixels)
    image_stack.append(cropped_array)
image_stack = np.array(image_stack)
print("nr_images, pixels_x, pixels_y = ", image_stack.shape)

# compute average image and plot
z_project = np.mean(image_stack, axis=0)

spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=10, threshold=50)
y = spots_LoG[:, 0]
x = spots_LoG[:, 1]
rois_array = np.column_stack((x, y))
num_rois = len(rois_array)

plt, ax = plt.subplots()
ax.imshow(z_project)
ax.scatter(x,y, marker='x', color='r')
plt.show()

# %% read counts within each ROI

def read_counts_within_rois(rois_array, image_stack):
    num_rois = rois_array.shape[0]
    num_images = image_stack.shape[0]
    
    # Initialize a matrix to store counts with shape (num_rois, num_images)
    counts_matrix = np.zeros((num_rois, num_images), dtype=int)
    
    for image_index in range(num_images):
        image = image_stack[image_index]
        
        for roi_index, roi in enumerate(rois_array):
            x, y = roi

            if rois_radius > 0:
                roi_region = image[int(y - rois_radius):int(y + rois_radius), int(x - rois_radius):int(x + rois_radius)]
            else:
                roi_region=image[x,y]

            counts = np.sum(roi_region)
            counts_matrix[roi_index, image_index] = counts
    
    return counts_matrix


counts_matrix = read_counts_within_rois(rois_array, image_stack)
counts_matrix = np.array(counts_matrix)
print("nr ROIs, nr counts = ", counts_matrix.shape)

# %% plot histograms

# Create a figure and axes for the subplots
fig, axes = plt.subplots(num_rois, 1, figsize=(10, 10))

# Plot histograms for each ROI
for i in range(num_rois):
    ax = axes[i]
    ax.hist(counts_matrix[i], bins=20, edgecolor='black')
    ax.set_title(f'Histogram of Counts for ROI {i+1}')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
