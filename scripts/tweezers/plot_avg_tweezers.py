# author; Marijn Venderbosch
# October 2024 - April 2025

#%%

import os
import numpy as np
import matplotlib.pyplot as plt

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from camera_image_class import CameraImage
from plotting_class import Plotting

# import image sequence
raw_data_path = 'Z:\\Strontium\\Images\\2025-04-17\\scan131340\\'
raw_data_suffix = 'image'
image_stack = CameraImage().import_image_sequence(raw_data_path, raw_data_suffix)

#%% variables

magnification = 1.25*50/4
pixel_size = 13e-6
bin_factor = 1
um = 1e-6 # m

#%%

z_project = np.mean(image_stack, axis=0)
pixels_x = z_project.shape[0]
roi_size_x = CameraImage.pixels_to_m(pixels_x, magnification, pixel_size,bin_factor)

pixels_y = z_project.shape[1]
roi_size_y = CameraImage.pixels_to_m(pixels_y, magnification, pixel_size, bin_factor)

# %% plotting'

fig1, ax1 = plt.subplots()
ax1.imshow(z_project, cmap='gist_yarg', extent=[0, roi_size_x/um, 0, roi_size_y/um])
ax1.set_xlabel('x (um)')
ax1.set_ylabel('y (um)')
ax1.tick_params(axis='both', direction='in')  

Plotting().savefig('output', 'tweezers_avg_image.png')
# %%
