# marijn venderbosch
# november 2023
"""script for analyzing time of flight data"""

#%% import modules

# standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy.constants

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

# user defined functions
from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage

# %% parameters

cam_mag = 0.8
pix_size = 3.45e-6  # [m]
bin_size = 4  # [pixels]

# %% functions 


def fit_and_return_parameters(xy, data, yguess):
    """fit data and return parameters"""

    # Initial guess for the parameters, bounds
    initial_guess = (30, 300, yguess, 10, 10, np.min(data))
    bounds = (0, [400, 1000, 1000, 50, 50, np.inf])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, covariance = curve_fit(FittingFunctions.gaussian_2d, xy, data, 
        p0 = initial_guess, bounds = bounds)
    
    # Calculate standard errors from the covariance matrix
    standard_errors = np.sqrt(np.diag(covariance))

    # Create a DataFrame to store parameters and standard errors
    result = pd.DataFrame({
        'Parameter': ['Amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'Offset'],
        'Value': params,
        'Standard Error': standard_errors
    })
    return result


def analyze_folder(folder_path, first_datapoint_ms, yguess):
    """for a given folder, export the image data and fit to extract parameters"""

    # Get a list of image files in the folder
    # only get the background subtracted ...fluor.tif
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(r'fluor.tif')])

    # Create a list to store individual DataFrames for each image
    parameters_list = []

    # Precompute meshgrid getting image size from first image in the list (image_files[0])
    original_image = CameraImage.load_image_from_file(folder_path, image_files[0])
    x_max = y_max = 0
    x_max = max(x_max, original_image.shape[1])
    y_max = max(y_max, original_image.shape[0])
    x = np.arange(0, x_max, 1)
    y = np.arange(0, y_max, 1)
    x, y = np.meshgrid(x, y)

    # empty lists to fill later
    sigmas_x = []
    d_sigmas_x = []
    sigmas_y = []
    d_sigmas_y = []
    tofs = []

    for idx, image_file in enumerate(image_files):
        # load image and flatten for fit
        original_image = CameraImage.load_image_from_file(folder_path, image_file)
        data = original_image.flatten()

        # Fit and get parameters for the entire image
        fitted_params_df = fit_and_return_parameters(np.vstack((x.flatten(), y.flatten())), data, yguess)

        # Add 'time_of_flight' parameter to the DataFrame, from ms to s
        tof_ms = (idx + first_datapoint_ms)*1e-3
        tofs.append(tof_ms)

        # Compute average sigma in pixels and convert to meter for x,y separately
        sigmas_x_px = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_x', 'Value'].values[0]
        sigma_x = CameraImage.pixels_to_m(sigmas_x_px, cam_mag, pix_size, bin_size)
        sigmas_x.append(sigma_x)

        sigmas_y_px = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_y', 'Value'].values[0]
        sigma_y = CameraImage.pixels_to_m(sigmas_y_px, cam_mag, pix_size, bin_size)
        sigmas_y.append(sigma_y)
 
        # obtain error in average sigma for x, y separately
        err_sigma_x_px = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_x', 'Standard Error'].values[0]
        d_sigma_x = CameraImage.pixels_to_m(err_sigma_x_px, cam_mag, pix_size, bin_size)
        d_sigmas_x.append(d_sigma_x)

        err_sigma_y_px = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_y', 'Standard Error'].values[0]
        d_sigma_y = CameraImage.pixels_to_m(err_sigma_y_px, cam_mag, pix_size, bin_size)
        d_sigmas_y.append(d_sigma_y)

        print('fitting ' + str(image_file))
        print('sigma_x = ' + str(np.round(sigma_x*1e6, 1)) + ' +/- '
              + str(np.round(d_sigma_x*1e6, 1)) + ' um') 
        print('sigma_y = ' + str(np.round(sigma_y*1e6, 1)) + ' +/- '
              + str(np.round(d_sigma_y*1e6, 1)) + ' um') 
        
        # Convert the individual DataFrame to a list
        parameters_list.append(fitted_params_df)
    
    # Convert 'sigma' column to a NumPy array
    sigmas_x = np.array(sigmas_x)
    d_sigmas_x = np.array(d_sigmas_x)
    sigmas_y = np.array(sigmas_y)
    d_sigmas_y = np.array(d_sigmas_y)
    tofs = np.array(tofs)
    print('fitted all images')

    return sigmas_x, d_sigmas_x, sigmas_y, d_sigmas_y, tofs


def compute_temp_tof(tof_array, sigmas_array, error_bars):
    """fit data and return parameters"""

    # square x,y so we can do a linear fit
    t_squared = tof_array**2
    sigmas_squared = sigmas_array**2

    # fit the data including errorbars
    params, covariance = curve_fit(FittingFunctions.linear_func, 
        xdata = t_squared, ydata= sigmas_squared, sigma = error_bars)

    # get sigma^2(t=0) from y-intersection point
    sigma0 = np.sqrt(params[0])

    # extract temperature from slope
    sr_mass = 88*scipy.constants.proton_mass
    Boltzmann = scipy.constants.Boltzmann

    # get temperature from slope
    slope = params[1]
    temperature = slope*sr_mass/Boltzmann

    # get standard errors from covariance matrix
    standard_errors = np.sqrt(np.diag(covariance))
    err_sigm0 = standard_errors[0]
    error_temp = standard_errors[1]*sr_mass/Boltzmann
    return params, sigma0, temperature, err_sigm0, error_temp


def main(folder, first_datapoint_ms, yguess):
    # fit each image in the specified folder directly with 2d gaussians
    sigmas_x, d_sigmas_x, sigmas_y, d_sigmas_y, tof_array = analyze_folder(
        folder, first_datapoint_ms, yguess)

    # compute error bars
    error_bars_x = 2*sigmas_x*d_sigmas_x
    error_bars_y = 2*sigmas_y*d_sigmas_y

    # fit the sigmas as a function of time of flight to extract temperature
    # at the same time also compute sigma(t=0) by extrapolating to t=0
    params_x, sigma0_x, temp_x, sigma0_error_x, temp_error_x = compute_temp_tof(
        tof_array, sigmas_x, error_bars_x)
    params_y, sigma0_y, temp_y, sigma0_error_y, temp_error_y = compute_temp_tof(
        tof_array, sigmas_y, error_bars_y)

    # print result
    print(f"Tx = {np.round(temp_x*1e6, 2)} +/- {np.round(temp_error_x*1e6, 2)} uK")
    print(f"Ty = {np.round(temp_y*1e6, 2)} +/- {np.round(temp_error_y*1e6, 2)} uK")
    print(f"sigma_x(t=0) = {np.round(sigma0_x*1e6, 2)} +/- {np.round(sigma0_error_x*1e6, 2)} um")
    print(f"sigma_y(t=0) = {np.round(sigma0_y*1e6, 2)} +/- {np.round(sigma0_error_y*1e6, 2)} um")

    fig, ax = plt.subplots()

    # plot datapoints and error bars separately for x
    ax.scatter(tof_array**2, sigmas_x**2, label=r'$\sigma_x$', marker='o')
    ax.errorbar(tof_array**2, sigmas_x**2,
        yerr = error_bars_x, linestyle='', color='black', capsize=3, markersize=5)
    
    # and y
    ax.scatter(tof_array**2, sigmas_y**2, label=r'$\sigma_y$', marker='o')
    ax.errorbar(tof_array**2, sigmas_y**2,
        yerr = error_bars_y, linestyle='', color='black', capsize=3, markersize=5)

    # x axis for linear fit
    t2_plot_arr = np.linspace(np.min(tof_array**2), np.max(tof_array)**2, 100)

    # plot linear fit Tx
    ax.plot(t2_plot_arr, FittingFunctions.linear_func(t2_plot_arr, *params_x),
        label='linear Fit: Tx = ' + str(np.round(temp_x*1e6, 2)) + r' $\pm$ ' + 
        str(np.round(temp_error_x*1e6, 2)) + ' uK', color='blue')
    
    # plot linear fit Ty
    ax.plot(t2_plot_arr, FittingFunctions.linear_func(t2_plot_arr, *params_y),
        label='linear Fit: Ty = ' + str(np.round(temp_y*1e6, 2)) + r' $\pm$ ' + 
        str(np.round(temp_error_y*1e6, 2)) + ' uK', color='orange')
    
    ax.set_xlabel(r'$t^2$ [s${}^2$]')
    ax.set_ylabel(r'$\sigma(t)^2$ [m${}^2$]')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    folder_path = r'T:\\KAT1\\Marijn\\redmot\\time of flight\\nov15measurements\\varying intensity\37741\\'

    # first time of flight image time in ms
    first_datapoint = 1  # ms

    # starting guess for y value of 2D gaussian. Vary if your fit fails
    # because the S/N drops for later times, probably good idea to match roughly y position
    # in the later fits
    y_guess = 200

    main(folder_path, first_datapoint_ms = first_datapoint, yguess = y_guess)
