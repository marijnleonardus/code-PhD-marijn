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
import imageio.v2 as imageio 

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
    initial_guess = (30, 300, yguess, 10, 10, 0, np.min(data))
    bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, covariance = curve_fit(FittingFunctions.gaussian_2d_angled, xy, data, 
        p0=initial_guess, bounds=bounds)
    
    # Calculate standard errors from the covariance matrix
    standard_errors = np.sqrt(np.diag(covariance))

    # Create a DataFrame to store parameters and standard errors
    result = pd.DataFrame({
        'Parameter': ['Amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'Offset'],
        'Value': params,
        'Standard Error': standard_errors
    })
    return result


def compute_average_sigma(sigma_x, sigma_y, magnification, binning, pixel_size):
    """Calculate the average sigma in pixels"""

    avg_sigma_pixels = 0.5*(sigma_x + sigma_y)

    # Convert sigma from pixels to meters
    sigma_meters_image = avg_sigma_pixels*binning*pixel_size

    # convert pixels to meters based on magnification
    sigma_meters_object = sigma_meters_image/magnification
    return sigma_meters_object


def analyze_folder(folder_path, first_datapoint_ms, yguess):
    """for a given folder, export the image data and fit to extract parameters"""

    # Get a list of image files in the folder
    # only get the background subtracted ...fluor.tif
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(r'fluor.tif')])

    # Create a list to store individual DataFrames for each image
    parameters_list = []

    # Precompute meshgrid
    original_image = CameraImage.load_image_from_file(folder_path, image_files[0])
    x_max = y_max = 0
    x_max = max(x_max, original_image.shape[1])
    y_max = max(y_max, original_image.shape[0])

    x = np.arange(0, x_max, 1)
    y = np.arange(0, y_max, 1)

    # empty lists to fill later
    x, y = np.meshgrid(x, y)
    sigmas = []
    err_sigmas = []
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

        # Compute average sigma in meters
        sigmas_x = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_x', 'Value'].values[0]
        sigmas_y = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_y', 'Value'].values[0]
        avg_sigma_meters = compute_average_sigma(sigmas_x, sigmas_y, cam_mag, bin_size, pix_size)

        # obtain error in average sigma
        error_sigma_x = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_x', 'Standard Error'].values[0]
        error_sigma_y = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_y', 'Standard Error'].values[0]
        avg_err_sigma_meters = compute_average_sigma(error_sigma_x, error_sigma_y, cam_mag, bin_size, pix_size)
        print('sigma = ' + str(np.round(avg_sigma_meters*1e6, 1)) + ' +/- '
              + str(np.round(avg_err_sigma_meters*1e6, 1)) + ' um')

        # Convert the individual DataFrame to a list
        parameters_list.append(fitted_params_df)
        sigmas.append(avg_sigma_meters)
        err_sigmas.append(avg_err_sigma_meters)

    # Convert 'sigma' column to a NumPy array
    sigmas = np.array(sigmas)
    err_sigmas = np.array(err_sigmas)
    tofs = np.array(tofs)
    return sigmas, err_sigmas, tofs


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
    import scipy.constants
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
    sigmas, sigmas_error, tof_array = analyze_folder(folder, first_datapoint_ms, yguess)
    
    # compute errorbars
    error_bars = 2*sigmas*sigmas_error
    
    # fit the sigmas as a function of time of flight to extract temperature
    # at the same time also compute sigma(t=0) by extrapolating to t=0
    params, sigma0, temp, sigma0_error, temp_error = compute_temp_tof(tof_array, sigmas, error_bars)
    
    print(f"T = {np.round(temp*1e6, 2)} +/- {np.round(temp_error*1e6, 2)} uK")
    print(f"sigma = {np.round(sigma0*1e6, 2)} +/- {np.round(sigma0_error*1e6, 2)} um")
    
    # plot data points and errobars separately 
    fig, ax = plt.subplots()
    ax.scatter(tof_array**2, sigmas**2, label='datapoints', marker='o')
    ax.errorbar(tof_array**2, sigmas**2,
        yerr=error_bars, linestyle='', color='black', capsize=3, markersize=5)

    # plot a linear fit of the data
    t_squared_plotarray = np.linspace(np.min(tof_array**2), np.max(tof_array)**2, 100)
    ax.plot(t_squared_plotarray, linear_func(t_squared_plotarray, *params),
        label='linear Fit: T = ' + str(np.round(temp*1e6, 2)) + r' $\pm$ ' + 
        str(np.round(temp_error*1e6, 2)) + ' uK', color='red')
    
    ax.set_xlabel(r'$t^2$ [s${}^2$]')
    ax.set_ylabel(r'$\sigma(t)^2$ [m${}^2$]')
    ax.legend()
    plt.show()


if __name__ == "__main__":
   folder_path = r'T:\\KAT1\\Marijn\\redmot\\time of flight\\nov15measurements\\varying time\37898\\'
   
   # first time of flight image time in ms
   first_datapoint = 1  # ms

   # starting guess for y value of 2D gaussian. Vary if your fit fails
   y_guess = 160

   main(folder_path, first_datapoint_ms = first_datapoint, yguess = y_guess)
