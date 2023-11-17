# marijn venderbosch
# november 2023
"""script for analyzing time of flight data"""

 # %% imports

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import imageio.v2 as imageio 
from scipy.constants import proton_mass, Boltzmann

# %% variables

sr_mass = 88*proton_mass
magnn = 100/125

# %%

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """2D Gaussian function.

    Parameters:
    - xy: 2D array containing x and y coordinates.
    - amplitude: Amplitude of the Gaussian.
    - xo: x-coordinate of the center.
    - yo: y-coordinate of the center.
    - sigma_x: Standard deviation along the x-axis.
    - sigma_y: Standard deviation along the y-axis.
    - theta: Rotation angle of the Gaussian in radians.
    - offset: Offset or background.

    Returns:
    - Values of the Gaussian function evaluated at given coordinates.
    """
    x, y = xy
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    
    # Gaussian function
    g = amplitude * np.exp(-(a * (x - xo)**2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo)**2)) + offset
    return g


def fit_and_return_parameters(xy, data):
    """fit data and return parameters"""

    # Initial guess for the parameters
    initial_guess = (30, 300, 150, 10, 10, 0, np.min(data))
    
    # Define bounds for the parameters, including the constraint for theta
    bounds = ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, _ = curve_fit(gaussian_2d, xy, data, p0=initial_guess, bounds=bounds)
    return pd.Series(params, index=['Amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'Offset'])


def compute_average_sigma(sigma_x, sigma_y, magnification, binning=4, pixel_size=3.45e-6):
    """Calculate the average sigma in pixels"""

    avg_sigma_pixels = 0.5*(sigma_x + sigma_y)

    # Convert sigma from pixels to meters
    sigma_meters_image = avg_sigma_pixels*binning*pixel_size

    # convert pixels to meters based on magnification
    sigma_meters_object = sigma_meters_image/magnification
    return sigma_meters_object


# %%

def analyze_folder(folder_path):
    """for a given folder, export the image data and fit to extract parameters"""

    # Get a list of image files in the folder
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('fluor.tif')])

    # Create a list to store individual DataFrames for each image
    parameters_list = []

    # Precompute meshgrid
    x_max, y_max = 0, 0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        original_image = imageio.imread(image_path)
        x_max = max(x_max, original_image.shape[1])
        y_max = max(y_max, original_image.shape[0])

    x = np.arange(0, x_max, 1)
    y = np.arange(0, y_max, 1)

    # empty lists to fill later
    x, y = np.meshgrid(x, y)
    sigmas = []
    tofs = []

    for idx, image_file in enumerate(image_files):
        # Get the full image path
        image_path = os.path.join(folder_path, image_file)

        # Read the image using imageio
        original_image = imageio.imread(image_path)

        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            original_image = np.mean(original_image, axis=2).astype(np.uint8)

        # Flatten the entire image
        data = original_image.flatten()

        # Fit and get parameters for the entire image
        fitted_params = fit_and_return_parameters(np.vstack((x.flatten(), y.flatten())), data)

        # Add 'time_of_flight' parameter to the DataFrame, from ms to s
        tof_ms = (idx + 1) * 1e-3
        tofs.append(tof_ms)

        # Compute average sigma in meters
        avg_sigma_meters = compute_average_sigma(fitted_params['sigma_x'], fitted_params['sigma_y'], magnification=magn)
        print("Average Sigma (meters):", avg_sigma_meters)

        # Convert the individual DataFrame to a list
        parameters_list.append(fitted_params)

        sigmas.append(avg_sigma_meters)

    # Convert 'sigma' column to a NumPy array
    sigmas_array = np.array(sigmas)
    tofs_array = np.array(tofs)
    return tofs_array, sigmas_array


folder_path = r"T:\KAT1\Marijn\redmot\time of flight\nov15measurements\37723"
tof_array, sigmas_fitted = analyze_folder(folder_path)

# %% plotting and fitting

def linear_func(x, a, b):
    return a + b*x

def fit_tof(tof_array, sigmas_array):
    """fit data and return parameters"""

    t_squared = tof_array**2
    sigmas_squared = sigmas_array**2
    popt, _ = curve_fit(linear_func, t_squared, sigmas_squared)
    return popt

t_squared_plotarray = np.linspace(np.min(tof_array**2), np.max(tof_array)**2, 100)

popt = fit_tof(tof_array, sigmas_fitted)
slope = popt[0]
temperature_kelvin = slope*sr_mass/Boltzmann
temperature_uk = temperature_kelvin*1e6
print(temperature_uk)

fig, ax = plt.subplots()
ax.scatter(tof_array**2, sigmas_fitted**2, label='Time of Flight (squared)')
ax.plot(t_squared_plotarray, linear_func(t_squared_plotarray, *popt), label='Linear Fit', color='red')
ax.set_xlabel(r'$t^2$ [s${}^2$]')
ax.set_ylabel(r'$\sigma(t)^2$ [m${}^2$]')
ax.legend()
plt.show()










#if __name__ == "__main__":
#    folder_path = r"T:\KAT1\Marijn\redmot\time of flight\nov14measurements\33095"
#    main(folder_path)


# %%
