# marijn venderbosch
# november 2023

# standard libraries
import os
import numpy as np

# append modules dir
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
print(modules_dir)
sys.path.append(modules_dir)

# import user module
from image_analysis_class import SpotDetectionFitting
from camera_image_class import CameraImage

# variables
folder_path = r"Z://Strontium//Images//2024-10-21//442187//"
image_name = r"0000absorption.tif"
magnification = 150/250
pixel_size = 3.45e-6  # [m]


def compute_cross_section(wavelength):
    """compute cross section

    Args:
        wavelength (float): wavelength in [m]

    Returns:
        cross_section (float): cross section in [m^2]
    """
    cross_section = 3*wavelength**2/(2*np.pi)
    return cross_section


def main(method):
    """
    Main function to compute the number of atoms from an image using two methods: 
    pixel count and Gaussian fit.

    Args:
        method (str): Method to compute atom number. Options are "pixel_count" 
                      or "gaussian_fit".
    
    This function performs the following steps:
    1. Loads an image from a specified file path.
    2. Creates an ImageAnalysis object to analyze the image.
    3. Computes the atomic cross section.
    4. Depending on the chosen method, the function either:
       - Computes the total pixel count, converts it to optical density, 
         integrates it, and calculates the atom number.
       - Fits a 2D Gaussian to the image, extracts fit parameters, and calculates
         the atom number.
    5. Prints the computed atom number.
    """
    os.system('cls')
    raw_data = CameraImage.load_image_from_file(folder_path, image_name)

    # Create image analysis object from class 
    ImageAnalysis = SpotDetectionFitting(sigma=40, threshold_detection=0.0007, image=raw_data)
   
    cross_section = compute_cross_section(461e-9)
    
    if method == "pixel_sum":
        # compute atom number from pixel count
        signal_px_count = ImageAnalysis.total_pixel_count(window_radius=50, print_enabled=True, plot_enabled=True)

        # the amplitude was multiplied by 1000 in the analyze function of artiq. So divide by this to get OD
        total_od_count = signal_px_count/1000

        # multiply by px size to ingrate over x,y instead of px_x, px_y. Take into account magnification
        integrated_od = total_od_count*(pixel_size/magnification)**2
        atom_nr = integrated_od/cross_section
    
    if method == "gaussian_fit":
        # compute atom number from 2d image 
        amplitude_px, sigma_x_px, sigma_y_px = ImageAnalysis.twod_gaussian_fit(
            amplitude_guess=200, offset_guess=200, print_enabled=False, plot_enabled=False)

        # convert sigma from pixels to meters
        sigma_x = CameraImage.pixels_to_m(sigma_x_px, magnification, pixel_size, bin_size=1)
        sigma_y = CameraImage.pixels_to_m(sigma_y_px, magnification, pixel_size, bin_size=1)
        
        # divide by 1000 because we multiplied by 1000 in the analyze function of artiq
        amplitude_od = amplitude_px/1000 
        atom_nr = 2*np.pi*amplitude_od*sigma_x*sigma_y/cross_section

        # compute density. We don't know sigma in the z direction. Take average of x and y
        sigma_z = (sigma_x + sigma_y)/2
        atomic_density = atom_nr/((2*np.pi)**1.5*sigma_x*sigma_y*sigma_z)
        atomic_density_cm3 = atomic_density/(100)**3
        print("atomic density: " , f"{atomic_density_cm3:.0e}", " cm^-3")
    
    print("computed atom number: " , f"{atom_nr:.0e}")


if __name__ == '__main__':
    main(method="gaussian_fit")
