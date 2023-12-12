# author: Marijn Venderbosch
# December 2022

"""
Script makes a plot of the laser induced fluorescence from the MOT with
a color overlay and a scalebar
Also exports sum over rows and columms and fits a Voigt profile through it

The script makes use of gridspec, which seperates the plotting regino in boxes
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


# %% Variables

# gridspec setup
w = 330
h = 20000

# %% functions


def main(folder, file_name, color):
     # load image
    image = CameraImage.load_image_from_file(folder, file_name)
        
    image_cropped = CameraImage.crop_center(image, crop_r, crop_r)
    max_nr_pixel_counts = np.max(image_cropped)

    # compute histograms
    hist_rows, hist_cols = CameraImage.compute_histograms_x_y(image_cropped)

    # pixel arrays used to number pixels in cropped image
    pixels_x = CameraImage.return_pixel_array(crop_r)
    pixels_y = CameraImage.return_pixel_array(crop_r)

    # guess for fittin data
    fitting_guess = [0.2, 0.8, 40, 16]  # offset, amplitude, middle, width

    # fit data
    popt_rows, _ = curve_fit(FittingFunctions.gaussian_function, pixels_y, hist_rows, p0=fitting_guess)
    popt_cols, _ = curve_fit(FittingFunctions.gaussian_function, pixels_x, hist_cols, p0=fitting_guess)

    # print sigma_x,y
    sigma_x_um = CameraImage.convert_pixels_to_distance(popt_cols, 'x', pixel_size, cam_mag)
    sigma_y_um = CameraImage.convert_pixels_to_distance(popt_rows, 'y', pixel_size, cam_mag)
    print(sigma_x_um)
    print(sigma_y_um)

    # Initialize gridspec with correct ratios
    fig = plt.figure(figsize=(5.1, 4))

    gs = gridspec.GridSpec(4, 4, hspace=0, wspace=0, figure=fig, 
        height_ratios=[1, h, 1, h/4.8], width_ratios=[1, w, 1, w/4.28])

    ax1 = plt.subplot(gs[0:3, 0:3])
    ax2 = plt.subplot(gs[1, 3])
    ax3 = plt.subplot(gs[3, 1])
    ax4 = plt.subplot(gs[3, 3])

    img = ax1.imshow(image_cropped, interpolation='nearest', origin='lower', vmin=0.)
    img.set_cmap(color)
    ax1.axis('off')

    # Scalebar, nr of pixels has to be integer
    scalebar_object_size = 1e-3  # m
    scalebar_pixels = int(scalebar_object_size/(pixel_size/cam_mag))

    scale_bar = AnchoredSizeBar(ax1.transData,
        scalebar_pixels,  # pixels
        r'1 mm',  # real life distance of scale bar
        'upper left', pad=0, color='black', frameon=False, size_vertical=2.5)

    # plot sum over rows, as well as a guassian fit
    ax2.scatter(-np.flip(hist_rows), pixels_y * pixel_size / cam_mag * 10e2, s=5)
    ax2.plot(-np.flip(FittingFunctions.gaussian_function(pixels_y, *popt_rows)),
        pixels_y * pixel_size/cam_mag * 10e2, color='r', linewidth=1)

    # plot sum over columns as well as gaussian fit
    ax3.scatter(pixels_x * pixel_size/cam_mag * 10e2, hist_cols, s=5)
    ax3.plot(-pixels_x * pixel_size/cam_mag*10e2, 
        FittingFunctions.gaussian_function(pixels_x, *popt_cols), color='r', linewidth=1)

    ax2.set_ylabel(r'$y$ [mm]')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.set_ticks_position('right')
    ax2.set_xticks([])
    ax3.set_xlabel(r'$x$ [mm]')
    ax3.set_yticks([])
    ax4.axis('off')

    savefig()

    ## ATOM NUMBER STUFF
    # obtain atom number
    atoms_mot = NumberAtoms.atom_number_from_image(popt_cols, popt_rows, max_nr_pixel_counts,
        camera_gain, exposure_time, photon_percount, trans_const, gamma_461, lens_distance, lens_radius)

    # obtain atoms per cubic cm
    atom_density = NumberAtoms.atomic_density_from_atom_number(
        atoms_mot, popt_cols[3], popt_rows[3])
    
    return atoms_mot, atom_density


def savefig():
    export_folder = 'exports/'
    export_name = 'mot_fluoresecence_fit.pdf'
    export_location = export_folder + export_name
    plt.savefig(export_location, dpi = 300, pad_inches = 0, bbox_inches = 'tight') 
    plt.show()


if __name__ == '__main__':
    # MOT plot params
    cam_mag = 0.8    
    color_plot = 'Reds'

    # atom nr parameters
    gamma_461 = 2*pi*32e6  # Hz (32 MHz)
    crop_r = 198  # pixels
    pixel_size = 6.5e-6  # microns
    lens_radius = 25e-3  # m
    lens_distance = 20e-2  # m
    trans_const = 0.6
    photon_percount = 6e3  
    camera_gain = 1
    exposure_time = 10e-3  # s

    # importing data
    folder = r'T:\KAT1\Marijn\redmot\redmot_bb'
    file_name = r'\0002_crop.tif'

    atoms_mot, atom_density = main(folder, file_name, color_plot)
    
    print('Number of atoms ~ ' + str(f"{Decimal(atoms_mot):.0E}"))
    print('Atom density ~ ' + str(f"{Decimal(atom_density):.0E}"))
