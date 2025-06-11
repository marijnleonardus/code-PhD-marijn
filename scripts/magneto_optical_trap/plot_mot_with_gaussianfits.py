# author: Marijn Venderbosch
# December 2022

"""
This script plots the laser-induced fluorescence from a magneto-optical trap
(MOT) with a color overlay and scalebar. It also exports the row and column sums
of the image and fits a Gaussian to each profile.

The plotting uses gridspec to allocate subregions.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.constants import pi

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

from camera_image_class import CameraImage
from fitting_functions_class import FittingFunctions
from plotting_class import Plotting
from image_analysis_class import ManipulateImage, AbsorptionImage

# Parameters
image_type = 'absorption'
w = 330             # GridSpec width ratio
h = 20000           # GridSpec height ratio
cam_mag = 0.6       # Camera magnification
pixel_size = 3.45e-6  # m
bin_size = 1
crop_r = 160      # Crop size in pixels
folder_name = r'T:\\KAT1\\Marijn\\thesis_measurements\\mot\\sf_mot\\absorption\\442187\\'
file_name = r'0000absorption.tif'

def main(image, cmap, show_gaussian_fit):
    """
    Fit Gaussian profiles to the image data and plot the fluorescence image,
    with an optional overlay of Gaussian fits along x and y.
    
    Parameters:
        image (np.ndarray): 2D image array.
        cmap (str): Colormap for the image.
        show_gaussian_fit (bool): Whether to overlay the Gaussian fits.
    """
    # --- 1. Compute and fit profiles ---
    # Sum over rows and columns
    hist_rows, hist_cols = CameraImage.compute_pixel_sums_x_y(image)
    
    # Define pixel coordinates for the cropped image (same for x and y)
    pixels = np.linspace(-crop_r/2, crop_r/2 - 1, crop_r)
    
    # Initial guess for [offset, amplitude, center, width]
    fit_guess = [0.1, 0.8, 0, 8]
    
    # Fit Gaussian functions to both profiles
    popt_rows, _ = curve_fit(FittingFunctions.gaussian_function, pixels, hist_rows, p0=fit_guess)
    popt_cols, _ = curve_fit(FittingFunctions.gaussian_function, pixels, hist_cols, p0=fit_guess)
    
    # Convert the Gaussian width (sigma) from pixels to meters
    sigma_x_px = popt_cols[3]
    sigma_z_px = popt_rows[3]
    sigma_x = CameraImage.pixels_to_m(sigma_x_px, cam_mag, pixel_size, bin_size)
    sigma_z = CameraImage.pixels_to_m(sigma_z_px, cam_mag, pixel_size, bin_size)
    print(f"{np.round(sigma_x*1e6)} μm")
    print(f"{np.round(sigma_z*1e6)} μm")
    
    # --- 2. Plot the MOT fluorescence image ---
    fig = plt.figure(figsize=(5.1, 4))
    gs = gridspec.GridSpec(
        4, 4, hspace=0, wspace=0, figure=fig,
        height_ratios=[1, h, 1, h/4.8],
        width_ratios=[1, w, 1, w/4.28]
    )

    # manipulate image
    if image_type == 'fluorescence':
        # normalize
        image = image/np.max(image)
    elif image_type == 'absorption':
        # compute OD essentially reversing the operation to avoid zero
        image = AbsorptionImage.compute_od(image)
    else:
        raise ValueError(f"Unknown image type: {image_type}")
    
    ax_img = plt.subplot(gs[0:3, 0:3])
    img_artist = ax_img.imshow(image, interpolation='nearest', origin='upper', vmin=0., aspect='equal')
    img_artist.set_cmap(cmap)
    ax_img.axis('off')

    # add colorbar
    fig.colorbar(img_artist, ax=ax_img)

    # Add scalebar (convert a 0.5 mm object to pixel units)
    scalebar = ScaleBar(pixel_size*bin_size/cam_mag, units='m', location='upper left')
    ax_img.add_artist(scalebar)

    # --- 3. Optionally overlay Gaussian fits ---
    if show_gaussian_fit:
        # Calculate axes (in meters) for the fit plots using the fitted centers
        center_x = popt_cols[2]
        center_y = popt_rows[2]
        axis_x = CameraImage.pixels_to_m(pixels - int(center_x), cam_mag, pixel_size, bin_size)
        axis_z = CameraImage.pixels_to_m(pixels - int(center_y), cam_mag, pixel_size, bin_size)
        
        # Vertical (row) profile subplot
        ax_fit_rows = plt.subplot(gs[1, 3])
        fitted_gaussian_rows = FittingFunctions.gaussian_function(pixels, popt_rows[0], popt_rows[1], 0, sigma_z_px)
        ax_fit_rows.scatter(-hist_rows, axis_z * 1e3, s=7)
        ax_fit_rows.plot(-fitted_gaussian_rows, axis_z * 1e3, color='r', linewidth=1)
        ax_fit_rows.set_ylabel(r'$y$ [mm]')
        ax_fit_rows.yaxis.set_label_position('right')
        ax_fit_rows.yaxis.set_ticks_position('right')
        ax_fit_rows.set_xticks([])
        
        # Horizontal (column) profile subplot
        ax_fit_cols = plt.subplot(gs[3, 1])
        # Use the horizontal fit parameters (popt_cols) here
        fitted_gaussian_cols = FittingFunctions.gaussian_function(pixels, popt_cols[0], popt_cols[1], 0, sigma_x_px)
        ax_fit_cols.scatter(axis_x * 1e3, hist_cols, s=7)
        ax_fit_cols.plot(-np.flip(axis_x * 1e3), fitted_gaussian_cols, color='r', linewidth=1)
        ax_fit_cols.set_xlabel(r'$x$ [mm]')
        ax_fit_cols.set_yticks([])
    

if __name__ == '__main__':
    # Load image and crop to the center
    image = CameraImage.load_image_from_file(folder_name, file_name)

    # needs to be cropped 
    # also for the gaussian fit the nr of pixels needs to be correct
    image = ManipulateImage.crop_to_region_of_interest(image, 470, 140, crop_r)

    # Fit the image and plot the result (set show_gaussian_fit True or False as needed)
    main(image, cmap='Reds', show_gaussian_fit=False)

    Plotting.savefig('output', file_name='\mot_' + image_type + '_plot.pdf')
