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
from utils.units import um

# import image sequence
raw_data_path = 'Z:\\Strontium\\Images\\2025-05-12\\scan143438\\'
raw_data_suffix = 'image'
image_stack = CameraImage().import_image_sequence(raw_data_path, raw_data_suffix)

#%% variables

f_obj = 4e-3 # m
f_emccd = 50e-3 # m
M_telescope = 1.25 # magnification of the telescope
magnification = M_telescope*f_emccd/f_obj # 1.25 is the magnification of the objective lens
pixel_size = 13e-6
bin_factor = 1

#%%

z_project = np.mean(image_stack, axis=0)
random_image = image_stack[36]

pixels_x = z_project.shape[0]
roi_size_x = CameraImage.pixels_to_m(pixels_x, magnification, pixel_size,bin_factor)

pixels_y = z_project.shape[1]
roi_size_y = CameraImage.pixels_to_m(pixels_y, magnification, pixel_size, bin_factor)

# %% plotting'

fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.subplots_adjust(wspace=0.5)  # Adjust horizontal space between subplots

ax1.imshow(random_image, cmap='gist_yarg', extent=[0, roi_size_x/um, 0, roi_size_y/um])
ax1.set_xlabel(r'x ($\mu$m)')
ax1.set_ylabel(r'y ($\mu$m)') 
ax1.tick_params(axis='both', direction='in')  

ax2.imshow(z_project, cmap='gist_yarg', extent=[0, roi_size_x/um, 0, roi_size_y/um])
ax2.set_xlabel(r'x ($\mu$m)')
ax2.set_ylabel(r'y ($\mu$m)')
ax2.tick_params(axis='both', direction='in')  

Plotting().savefig('output', 'tweezers_avg_image.png')
# %%
