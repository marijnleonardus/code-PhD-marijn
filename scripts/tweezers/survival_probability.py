# author: Marijn Venderbosch
# April 2025

import numpy as np
import os
import matplotlib.pyplot as plt

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries


# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'T:\\KAT1\\Marijn\scan174612\\selection'
binary_threshold = 15200

# load ROI counts from npy
# (nr ROIs, nr images)
roi_counts_matrix = np.load(os.path.join(images_path, 'roi_counts_matrix.npy'))
print("nr ROIs, nr images: ", np.shape(roi_counts_matrix))

# Perform binary thresholding: entries above threshold become 1, others become 0
binary_matrix = (roi_counts_matrix > binary_threshold).astype(int)

print(binary_matrix[:,2])
print(binary_matrix[:,3])

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
    
    # calculate avg over all ROIs (global)
    global_survival = np.nanmean(survival_matrix[:, im_idx])
    global_survival_matrix[im_idx] = global_survival

# Optional: save survival matrix
# np.save(os.path.join(images_path, 'survival_matrix.npy'), survival_matrix)

print(np.shape(global_survival_matrix))
plt.plot(global_survival_matrix, 'o-')
plt.show()