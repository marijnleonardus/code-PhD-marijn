# author: Marijn Venderbosch
# April 2025

"""first run the script calculate_roi_counts_plot_avg.py to create the roi_counts_matrix.npy file
then this script in combination with the binary_treshold computed from `histogram_and_threshold.py` 
will compute the survival probability of the atoms in the ROI"""

import numpy as np
import os
import pandas as pd

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from data_handling_class import reshape_roi_matrix

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
image_path = 'T:\\KAT1\\Marijn\scan174612\\'
binary_threshold = 15200

# load ROI counts from npy
# (nr ROIs, nr images)
roi_counts_matrix = np.load(os.path.join(image_path, 'roi_counts_matrix.npy'))
print("nr ROIs, nr images: ", np.shape(roi_counts_matrix))


def calculate_survival_probability(images_path, roi_counts_matrix, binary_threshold):
    """
    Calculate the survival probability of atoms in ROIs based on ROI counts matrix and binary threshold.
    Images 0, 3, 5, etc. are initial images, and images 1, 2, 4, etc. are final images.
    Then calculates surv probability for each pair of images.
    
    Parameters:
    - roi_counts_matrix (numpy.ndarray): Matrix containing ROI counts for each image.
    - binary_threshold (int): Threshold for binary classification of ROI counts.
    
    Returns:
    - survival_probability (numpy.ndarray): Survival matrix indicating survival status of atoms in ROIs.
    """

    # Perform binary thresholding: entries above threshold become 1, others become 0
    binary_matrix = (roi_counts_matrix > binary_threshold).astype(int)

    # Number of image pairs: floor divide by 2
    num_pairs = binary_matrix.shape[1] // 2

    # Initialize survival matrix with NaNs (undefined by default)
    # Using a floating-point array allows us to use np.nan
    survival_matrix = np.full((binary_matrix.shape[0], num_pairs), np.nan, dtype=float)

    # Process each pair of images
    for im_idx in range(num_pairs):
        initial = binary_matrix[:, 2*im_idx]     
        final = binary_matrix[:, 2*im_idx + 1]    
        
        # Create a mask for ROIs that had an atom initially
        # For ROIs where there was an atom, 1 = atom survived, 0 = atom disappeared
        mask = (initial == 1)
        survival_matrix[mask, im_idx] = final[mask]

    # reshape roi_counts_matrix depending on the number of averages
    # laod x_values. If multiple averages used x values contains duplicates
    df = pd.read_csv(images_path + 'log.csv')
    x_values, survival_matrix_sorted = reshape_roi_matrix(df, survival_matrix)

    # compute average by summing over repeated values
    survival_probability = np.nanmean(survival_matrix_sorted, axis=2)
    return x_values, survival_probability

x_values, survival_probability = calculate_survival_probability(image_path, roi_counts_matrix, binary_threshold)

# save survival matrix to be used by other scripts, as well as x_values vector
np.save(os.path.join(image_path, 'x_values.npy'), x_values)
np.save(os.path.join(image_path, 'survival_probability.npy'), survival_probability)
print("Nr ROIs, x_values: ", np.shape(survival_probability))
