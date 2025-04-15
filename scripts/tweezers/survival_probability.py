# author: Marijn Venderbosch
# April 2025
# %% 
import numpy as np
import os
import matplotlib.pyplot as plt
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
images_path = 'T:\\KAT1\\Marijn\scan174612\\'
binary_threshold = 15200

# %% load data 

# load ROI counts from npy
# (nr ROIs, nr images)
roi_counts_matrix = np.load(os.path.join(images_path, 'roi_counts_matrix.npy'))
print("nr ROIs, nr images: ", np.shape(roi_counts_matrix))

# Perform binary thresholding: entries above threshold become 1, others become 0
binary_matrix = (roi_counts_matrix > binary_threshold).astype(int)

# Number of image pairs: floor divide by 2
num_pairs = binary_matrix.shape[1] // 2

# Initialize survival matrix with NaNs (undefined by default)
# Using a floating-point array allows us to use np.nan
survival_matrix = np.full((binary_matrix.shape[0], num_pairs), np.nan, dtype=float)
global_survival_matrix = np.zeros(num_pairs, dtype=float)

# Process each pair of images
for im_idx in range(num_pairs):
    initial = binary_matrix[:, 2*im_idx]     
    final = binary_matrix[:, 2*im_idx + 1]    
    
    # Create a mask for ROIs that had an atom initially
    mask = (initial == 1)
    
    # For ROIs where there was an atom, 1 = atom survived, 0 = atom disappeared
    survival_matrix[mask, im_idx] = final[mask]

# Optional: save survival matrix
# np.save(os.path.join(images_path, 'survival_matrix.npy'), survival_matrix)

# reshape roi_counts_matrix depending on the number of averages
# laod x_values. If multiple averages used x values contains duplicates
df = pd.read_csv(images_path + 'log.csv')

x_values, survival_matrix_reshaped = reshape_roi_matrix(df, survival_matrix)
print(len(x_values))

print("nr ROIs, nr_avg, nr_x_values: ", np.shape(survival_matrix_reshaped))
nr_rois = survival_matrix_reshaped.shape[0]
nr_avg = survival_matrix_reshaped.shape[2]

plt.plot(survival_matrix_reshaped[0, :, 0], label='ROI 1')
plt.show()
# %%
