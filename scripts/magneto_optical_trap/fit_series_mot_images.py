# marijn venderbosch
# november 2023
"""script for analyzing time-of-flight data."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import proton_mass, Boltzmann

# Append path with 'modules' dir in parent folder
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage

# %% parameters
cam_mag = 0.85
pix_size = 3.45e-6  # [m]
bin_size = 4        # [pixels]
sr_mass = 88*proton_mass

folder_name = r'T:\\KAT1\\Marijn\\thesis_measurements\\mot\\sf_time_of_flight\\second try\\scan091510\\'
y_guess = 220  # Initial guess for y-center in the Gaussian fit

def fit_gaussian(xy, data, yguess):
    """Fit a 2D Gaussian to the image data and return the optimized parameters and errors.
    
    Parameters:
        xy (2xN np.array): Meshgrid coordinates flattened.
        data (1D np.array): Flattened image data.
        yguess (float): Initial guess for the y-center.
    
    Returns:
        popt (np.array): Fitted parameters [Amplitude, xo, yo, sigma_x, sigma_y, Offset].
        perr (np.array): Standard errors of the fitted parameters.
    """
    p0 = [30, 300, yguess, 10, 10, np.min(data)]
    bounds = (0, [400, 1000, 1000, 50, 50, np.inf])
    popt, pcov = curve_fit(FittingFunctions.gaussian_2d, xy, data, p0=p0, bounds=bounds)
    return popt, np.sqrt(np.diag(pcov))


def analyze_folder(folder_path, yguess, plot_gaussian_fits=False):
    """Process all images ending with 'fluor.tif' in a folder.
    
    Returns:
        sigmas_x, dsigmas_x, sigmas_y, dsigmas_y, tofs (all np.array):
            arrays of sigma values (in meters), their errors, and times-of-flight (in seconds).
    """
    # List and sort image files
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('fluorescence.tif')])
    if not image_files:
        raise ValueError("No 'fluor.tif' files found in the folder.")

    # Get image dimensions from the first image and create meshgrid coordinates
    first_img = CameraImage.load_image_from_file(folder_path, image_files[0])
    ny, nx = first_img.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    xy = np.vstack((x.flatten(), y.flatten()))

    # Lists to store results
    sigmas_x = []
    sigmas_y = []

    for idx, fname in enumerate(image_files):
        img = CameraImage.load_image_from_file(folder_path, fname)
        data = img.flatten()
        popt, _ = fit_gaussian(xy, data, yguess)

        # Parameters: [Amplitude, xo, yo, sigma_x, sigma_y, Offset]
        sigma_x_px, sigma_y_px = popt[3], popt[4]

        # Convert pixel sigma to meters
        sigma_x_m = CameraImage.pixels_to_m(sigma_x_px, cam_mag, pix_size, bin_size)
        sigma_y_m = CameraImage.pixels_to_m(sigma_y_px, cam_mag, pix_size, bin_size)

        sigmas_x.append(sigma_x_m)
        sigmas_y.append(sigma_y_m)

        print(f"Fitting {fname}")
        print(f"  sigma_x = {sigma_x_m*1e6:.0f} μm")
        print(f"  sigma_y = {sigma_y_m*1e6:.0f} μm")
        print("-" * 50)

        if plot_gaussian_fits:
            fig, ax = plt.subplots()
            # Reshape fitted model to image dimensions for plotting
            fitted = FittingFunctions.gaussian_2d((x, y), *popt).reshape(ny, nx)
            ax.imshow(fitted)
            ax.set_title(fname)
            plt.show()

    return np.array(sigmas_x), np.array(sigmas_y)


if __name__ == "__main__":
    size_x, size_y = analyze_folder(folder_name, y_guess)
    np.savetxt(folder_name + 'size_x.csv', size_x, delimiter=',')
    np.savetxt(folder_name + 'size_y.csv', size_y, delimiter=',')

    print("Done.")
    plt.show()
