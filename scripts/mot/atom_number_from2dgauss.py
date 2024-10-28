# marijn venderbosch
# november 2023
"""script for analyzing time of flight data"""

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


def main():
    os.system('cls')

    # load image from file
    raw_data = CameraImage.load_image_from_file(folder_path, image_name)

    # Create image analysis object from class and fit 2d gaussian
    ImageAnalysis = SpotDetectionFitting(sigma=60, threshold_detection=0.06, image=raw_data)
    amplitude_converted, sigma_x_px, sigma_y_px = ImageAnalysis.twod_gaussian_fit(
        amplitude_guess=200, offset_guess=200, plot_enabled=True)

    sigma_x = CameraImage.pixels_to_m(sigma_x_px, magnification, pixel_size, bin_size=1)
    sigma_y = CameraImage.pixels_to_m(sigma_y_px, magnification, pixel_size, bin_size=1)

    cross_section = compute_cross_section(461e-9)

    # compute atom number
    # the amplitude was multiplied by 1000 in the analyze function of artiq. So divide by this
    amplitude = amplitude_converted/1000
    atom_nr = 2*np.pi*amplitude*sigma_x*sigma_y/cross_section
    print("computed atom number: " , f"{atom_nr:.0e}")


if __name__ == '__main__':
    main()
