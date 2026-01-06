# author; Marijn Venderbosch
# October 2024 - April 2025

#%%

import os
import numpy as np
import matplotlib.pyplot as plt

# append path with 'modules' dir in parent folder

# add local modules
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from camera_image_class import CameraImage
from plot_utils import Plotting
from units import um

# import image sequence
raw_data_path = 'Z:\\Strontium\\Images\\2025-07-22\\scan181314\\'
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

Plot = Plotting('output')

figwidth=3.37*0.5 - 0.01
figheight=figwidth*0.61
fig1, ax1 = plt.subplots(figsize=(figwidth,figheight))

ax1.imshow(random_image, cmap='gist_yarg', extent=[0, roi_size_x/um, 0, roi_size_y/um])
ax1.set_xlabel(r'x ($\mu$m)')
ax1.set_ylabel(r'y ($\mu$m)') 
ax1.tick_params(axis='both', direction='in')  
Plot.savefig('tweezers_random_image.png')

fig2, ax2 = plt.subplots(figsize=(figwidth,figheight))
ax2.imshow(z_project, cmap='gist_yarg', extent=[0, roi_size_x/um, 0, roi_size_y/um])
ax2.set_xlabel(r'x ($\mu$m)')
ax2.set_ylabel(r'y ($\mu$m)')
ax2.tick_params(axis='both', direction='in')  
Plot.savefig('tweezers_avg_image.pdf')


# %%
