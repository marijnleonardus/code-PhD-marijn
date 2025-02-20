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
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage

# %% parameters
cam_mag = 0.8
pix_size = 3.45e-6  # [m]
bin_size = 4        # [pixels]
sr_mass = 88 * proton_mass


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


def analyze_folder(folder_path, first_datapoint_ms, yguess, plot_gaussian_fits=False):
    """Process all images ending with 'fluor.tif' in a folder.
    
    Returns:
        sigmas_x, dsigmas_x, sigmas_y, dsigmas_y, tofs (all np.array):
            arrays of sigma values (in meters), their errors, and times-of-flight (in seconds).
    """
    # List and sort image files
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('fluor.tif')])
    if not image_files:
        raise ValueError("No 'fluor.tif' files found in the folder.")

    # Get image dimensions from the first image and create meshgrid coordinates
    first_img = CameraImage.load_image_from_file(folder_path, image_files[0])
    ny, nx = first_img.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    xy = np.vstack((x.flatten(), y.flatten()))

    # Lists to store results
    sigmas_x, dsigmas_x = [], []
    sigmas_y, dsigmas_y = [], []
    tofs = []

    for idx, fname in enumerate(image_files):
        img = CameraImage.load_image_from_file(folder_path, fname)
        data = img.flatten()
        popt, perr = fit_gaussian(xy, data, yguess)
        # Parameters: [Amplitude, xo, yo, sigma_x, sigma_y, Offset]
        sigma_x_px, sigma_y_px = popt[3], popt[4]
        dsigma_x_px, dsigma_y_px = perr[3], perr[4]

        # Convert pixel sigma to meters
        sigma_x_m = CameraImage.pixels_to_m(sigma_x_px, cam_mag, pix_size, bin_size)
        sigma_y_m = CameraImage.pixels_to_m(sigma_y_px, cam_mag, pix_size, bin_size)
        dsigma_x_m = CameraImage.pixels_to_m(dsigma_x_px, cam_mag, pix_size, bin_size)
        dsigma_y_m = CameraImage.pixels_to_m(dsigma_y_px, cam_mag, pix_size, bin_size)

        # Time-of-flight in seconds (first_datapoint_ms is in ms)
        tof = (idx + first_datapoint_ms) * 1e-3
        tofs.append(tof)
        sigmas_x.append(sigma_x_m)
        dsigmas_x.append(dsigma_x_m)
        sigmas_y.append(sigma_y_m)
        dsigmas_y.append(dsigma_y_m)

        print(f"Fitting {fname}")
        print(f"  sigma_x = {sigma_x_m*1e6:.0f} ± {dsigma_x_m*1e6:.0f} μm")
        print(f"  sigma_y = {sigma_y_m*1e6:.0f} ± {dsigma_y_m*1e6:.0f} μm")
        print("-" * 50)

        if plot_gaussian_fits:
            fig, ax = plt.subplots()
            # Reshape fitted model to image dimensions for plotting
            fitted = FittingFunctions.gaussian_2d((x, y), *popt).reshape(ny, nx)
            ax.imshow(fitted)
            ax.set_title(fname)
            plt.show()

    return (np.array(sigmas_x), np.array(dsigmas_x),
            np.array(sigmas_y), np.array(dsigmas_y),
            np.array(tofs))


def compute_temp_tof(tof, sigma, dsigma):
    """Fit sigma^2 vs. t^2 data to extract initial cloud size and temperature.
    
    Parameters:
        tof (1D np.array): Time-of-flight (s).
        sigma (1D np.array): Cloud sizes (m).
        dsigma (1D np.array): Errors in cloud sizes (m).
    
    Returns:
        popt: Fitted parameters [sigma0^2, slope].
        sigma0: Cloud size at t=0 (m).
        temperature: Temperature (K).
        dsigma0: Uncertainty in sigma0.
        dtemp: Uncertainty in temperature.
    """
    t2 = tof**2
    sigma2 = sigma**2
    
    # Propagate error: Δ(sigma^2) ≈ 2*sigma*Δsigma
    error_bars = 2*sigma*dsigma

    popt, pcov = curve_fit(FittingFunctions.linear_func, t2, sigma2, sigma=error_bars)
    sigma0 = np.sqrt(popt[0])
    temperature = popt[1] * sr_mass / Boltzmann
    perr = np.sqrt(np.diag(pcov))
    return popt, sigma0, temperature, perr[0], perr[1] * sr_mass / Boltzmann


def main(folder, first_datapoint_ms, yguess, plot_gaussian_fits=False):
    # Analyze images and extract sigma values and their errors.
    sig_x, dsig_x, sig_y, dsig_y, tof = analyze_folder(folder, first_datapoint_ms, yguess, plot_gaussian_fits)

    # Fit sigma^2 vs. t^2 for x and y directions.
    popt_x, sigma0_x, temp_x, sigma0_err_x, temp_err_x = compute_temp_tof(tof, sig_x, dsig_x)
    popt_y, sigma0_y, temp_y, sigma0_err_y, temp_err_y = compute_temp_tof(tof, sig_y, dsig_y)

    print(f"Tx = {temp_x*1e6:.1f} ± {temp_err_x*1e6:.1f} μK")
    print(f"Ty = {temp_y*1e6:.1f} ± {temp_err_y*1e6:.1f} μK")
    print(f"sigma_x(t=0) = {sigma0_x*1e6:.1f} ± {sigma0_err_x*1e6:.1f} μm")
    print(f"sigma_y(t=0) = {sigma0_y*1e6:.1f} ± {sigma0_err_y*1e6:.1f} μm")

    # Plot sigma^2 data and fits.
    fig, ax = plt.subplots()
    t2 = tof**2
    ax.errorbar(t2, sig_x**2, yerr=2*sig_x*dsig_x, fmt='o', label=r'$\sigma_x^2(t)$')
    ax.errorbar(t2, sig_y**2, yerr=2*sig_y*dsig_y, fmt='o', label=r'$\sigma_y^2(t)$')

    t2_fit = np.linspace(t2.min(), t2.max(), 100)
    ax.plot(t2_fit, FittingFunctions.linear_func(t2_fit, *popt_x), 'b-', label=f'Fit: Tx = {temp_x*1e6:.1f}({temp_err_x*1e6:.1f}) μK')
    ax.plot(t2_fit, FittingFunctions.linear_func(t2_fit, *popt_y), 'r-', label=f'Fit: Ty = {temp_y*1e6:.1f}({temp_err_y*1e6:.1f}) μK')

    ax.set_xlabel(r'$t^2$ [s$^2$]')
    ax.set_ylabel(r'$\sigma^2$ [m$^2$]')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    folder_path = r'T:\KAT1\Marijn\redmot\time of flight\nov15measurements\varying detuning\\37815\\'
    first_datapoint = 1  # First image time-of-flight in ms
    y_guess = 170        # Initial guess for y-center in the Gaussian fit

    main(folder_path, first_datapoint_ms=first_datapoint, yguess=y_guess, plot_gaussian_fits=False)
