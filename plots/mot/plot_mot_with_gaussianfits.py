# author: Marijn Venderbosch
# December 2022

"""
Script makes a plot of the laser induced fluorescence from the MOT with
a color overlay and a scalebar
Also exports sum over rows and columms and fits a Gaussian

The script makes use of gridspec, which seperates the plotting regino in boxes

Estimates atom nr, this is currently not used
"""

# %% Imports

# standard libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from decimal import Decimal

# user defined functions
from modules.camera_image_class import CameraImage
from modules.fitting_functions_class import FittingFunctions
from modules.number_atoms_class import NumberAtoms
from modules.plotting_class import Plotting


# %% Variables

# gridspec setup
w = 330
h = 20000

# locations bb red mot
redmot_bb_folder = r'T:\KAT1\Marijn\redmot\redmot_bb'
redmot_bb_file_name = r'\0002_crop.tif'

# locations sf red mot
redmot_sf_folder = r'T:\KAT1\Marijn\redmot\redmot_sf'
redmot_sf_file_name = r'\sf_lowpower.tif'

# locations bb blue mot
bluemot_folder = r'T:\KAT1\Marijn\redmot\time of flight\nov15measurements\atom number\1_blue'
bluemot_filename = r'\0000.tif'

# %% functions


def main(image, color, show_gaussian_fit):
    """Fits Gaussians to image and plots the result
    
    Parameters:
    - image: numpy array (2d)
    - color: string
    - show_gaussian_fit: boolean, do you want to show the gaussian fits or not?"""
    
    # compute histograms by summing over rows and columns
    hist_rows, hist_cols = CameraImage.compute_histograms_x_y(image)

    # pixel arrays used to number pixels in cropped image
    pixels_x = pixels_z = np.linspace(-crop_r/2, crop_r/2 - 1, crop_r)

    # guess for fittin data
    fitting_guess = [0.2, 0.8, 0, 12]  # offset, amplitude, middle, width

    # fit pixel data
    popt_rows, _ = curve_fit(FittingFunctions.gaussian_function, pixels_z, hist_rows, p0=fitting_guess)
    popt_cols, _ = curve_fit(FittingFunctions.gaussian_function, pixels_x, hist_cols, p0=fitting_guess)

    # print sigma_x,y
    sigma_x_px = popt_cols[3]
    sigma_z_px = popt_rows[3]
    sigma_x = CameraImage.pixels_to_m(sigma_x_px, cam_mag, pixel_size, bin_size)
    sigma_z = CameraImage.pixels_to_m(sigma_z_px, cam_mag, pixel_size, bin_size)
    print(str(np.round(sigma_x*1e6)) + ' um')
    print(str(np.round(sigma_z*1e6)) + ' um')
    
    # Initialize gridspec with correct ratios
    fig = plt.figure(figsize=(5.1, 4))

    # grid spec. If show_guaussian_fit = False, only ax1 is used. 
    gs = gridspec.GridSpec(4, 4, hspace=0, wspace=0, figure=fig, 
        height_ratios=[1, h, 1, h/4.8], width_ratios=[1, w, 1, w/4.28])

    ax1 = plt.subplot(gs[0:3, 0:3])
    img = ax1.imshow(image, interpolation='nearest', origin='lower', vmin=0., aspect='equal')
    img.set_cmap(color)
    ax1.axis('off')

    # Scalebar, nr of pixels has to be integer
    scalebar_object_size = 0.5e-3  # m
    scalebar_pixels= CameraImage.m_to_pixels(scalebar_object_size, cam_mag, pixel_size, bin_size)
    scale_bar = AnchoredSizeBar(ax1.transData,
        scalebar_pixels,  # pixels
        r'0.5 mm',  # real life distance of scale bar
        'upper left', pad=0.5, color='black', frameon=False, size_vertical=2.5)
    ax1.add_artist(scale_bar)

    if show_gaussian_fit:
        # generate x,y axis true to scale for plot, displaced from center fit
        center_x = popt_cols[2]
        center_y = popt_rows[2]
        axis_x = CameraImage.pixels_to_m(pixels_x - int(center_x), cam_mag, pixel_size, bin_size)
        axis_z = CameraImage.pixels_to_m(pixels_z - int(center_y), cam_mag, pixel_size, bin_size)
        # make the subax for the gaussian fits
        ax2 = plt.subplot(gs[1, 3])
        ax3 = plt.subplot(gs[3, 1])

        # obtain fitted gaussian. Set center position to 0. Amplitude = 1 (normalized before)
        offset_z = popt_rows[0]
        amplitude_z = popt_rows[1]
        fitted_gaussian_rows = FittingFunctions.gaussian_function(pixels_z, offset_z, amplitude_z, 0, sigma_z_px)

        # plot sum over rows, as well as a guassian fit
        # y, x plot (reversed)
        # plot in mm (multiply 1e3)
        ax2.scatter(-(hist_rows), axis_z*1e3, s=7)
        ax2.plot(-fitted_gaussian_rows, axis_z*1e3, color='r', linewidth=1)

        # plot sum over columns as well as gaussian fit
        # plot in mm (multiply 1e3)
        offset_x= popt_rows[0]
        amplitude_x = popt_rows[1]
        fitted_gaussian_cols = FittingFunctions.gaussian_function(pixels_x, offset_x, amplitude_x, 0, sigma_x_px)
        ax3.scatter(axis_x*1e3, hist_cols, s=7)
        ax3.plot(-np.flip(axis_x*1e3), fitted_gaussian_cols, color='r', linewidth=1)

        ax2.set_ylabel(r'$y$ [mm]')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        ax2.set_xticks([])
        ax3.set_xlabel(r'$x$ [mm]')
        ax3.set_yticks([])

    Plotting.savefig(export_folder = 'exports/',
        file_name = 'mot_fluorescence_fit.pdf')

    """ ATOM NUMBER STUFF
    # obtain atom number
    atoms_mot = NumberAtoms.atom_number_from_image(popt_cols, popt_rows, max_nr_pixel_counts,
        camera_gain, exposure_time, photon_percount, trans_const, gamma_461, lens_distance, lens_radius)

    # obtain atoms per cubic cm
    atom_density = NumberAtoms.atomic_density_from_atom_number(
        atoms_mot, popt_cols[3], popt_rows[3])

    print('Number of atoms ~ ' + str(f"{Decimal(atoms_mot):.0E}"))
    print('Atom density ~ ' + str(f"{Decimal(atom_density):.0E}"))
    """


# %% execute function 
    

if __name__ == '__main__':
    # MOT plot params
    cam_mag = 0.8    
    color_plot = 'Blues'
    pixel_size = 3.45e-6  # m
    bin_size = 4
    crop_r = 150  # pixels

    # importing data
    folder = bluemot_folder #redmot_sf_folder
    file_name = bluemot_filename #redmot_sf_file_name

    # load image and crop to center
    image = CameraImage.load_image_from_file(folder, file_name)
    image_cropped = CameraImage.crop_center(image, crop_r, crop_r)

    # fit image and plot result
    main(image_cropped, color_plot, show_gaussian_fit=True)

    
    """# atom nr parameters
    gamma_461 = 2*pi*32e6  # Hz (32 MHz)
    lens_radius = 25e-3  # m
    lens_distance = 20e-2  # m
    trans_const = 0.6
    photon_percount = 6e3  
    camera_gain = 1
    exposure_time = 10e-3  # s

    
   """
