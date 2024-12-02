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
images_path = 'Z://Strontium/Images//2024-11-27//scan124532//'
file_name_suffix = 'image'  # import files ending with image.tif
show_plots = True
log_threshold = 10 # laplacian of gaussian kernel sensitivity
weight_center_pixel = 3
column_xvalues = 1 # column with independent variables

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
fig1.show()

# obtain counts over all ROIs
ROI = RoiCounts(weight_center_pixel, rois_radius)
rois_list, roi_counts_array = ROI.compute_pixel_sum_counts(images_list, y_coor, x_coor)

# laod detunings
file_name = images_path + 'log.csv'
df = pd.read_csv('Z://Strontium/Images//2024-11-27//scan124532//log.csv')
detunings = df.iloc[:, column_xvalues].to_numpy() # select right column, if mulitple averages used 

# compute number of averages per data point
detunings_unique = np.unique(detunings)
averages_per_xvalue = int(len(detunings)/len(detunings_unique))

# group detunings and counts together
grouped_counts = {}
for detuning, count in zip(detunings, roi_counts_array):
    grouped_counts.setdefault(detuning, []).append(count)

# Convert dictionary values to lists in the same order as the unique x-values
unique_detunings = sorted(grouped_counts.keys())
new_grouped_counts = [grouped_counts[detuning] for detuning in unique_detunings]

print("Unique x-values:", unique_detunings)
print("Grouped counts:", new_grouped_counts)

ROI.plot_average_of_roi(rois_list)
# compute average over all ROIs
# rois_array_3d = np.stack(rois_list, axis=0)
# average_image = np.mean(rois_array_3d, axis=0)
# fig2, ax2 = plt.subplots()
# ax2.imshow(average_image)

if show_plots == True:
    plt.show() 
