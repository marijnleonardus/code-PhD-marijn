# marijn venderbosch
# november 2023
"""script for analyzing time of flight data"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import imageio.v2 as imageio 


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
    initial_guess = (50, 300, 200, 20, 20, 0, np.min(data))
    
    # Define bounds for the parameters, including the constraint for theta
    bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, _ = curve_fit(gaussian_2d, xy, data, p0=initial_guess, bounds=bounds)

    # Create a DataFrame to store parameters and standard errors
    result = pd.DataFrame({
        'Parameter': ['Amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'Offset'],
        'Value': params
        })
    return result

def main(folder_path, image_name):
    image_path = os.path.join(folder_path, image_name)

    # Read the image using imageio
    original_image = imageio.imread(image_path)

    # Convert to grayscale if needed
    if len(original_image.shape) == 3:
        original_image = np.mean(original_image, axis=2).astype(np.uint8)

    # Precompute meshgrid
    x_max, y_max = 0, 0

    x_max = max(x_max, original_image.shape[1])
    y_max = max(y_max, original_image.shape[0])


    x = np.arange(0, x_max, 1)
    y = np.arange(0, y_max, 1)


    # empty lists to fill later
    x, y = np.meshgrid(x, y)        

    # Flatten the entire image
    data = original_image.flatten()

    # Fit and get parameters for the entire image
    fitted_params_df = fit_and_return_parameters(np.vstack((x.flatten(), y.flatten())), data)

    sigmas_x = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_x', 'Value'].values[0]
    sigmas_y = fitted_params_df.loc[fitted_params_df['Parameter'] == 'sigma_y', 'Value'].values[0]
    amplitude= fitted_params_df.loc[fitted_params_df['Parameter'] == 'Amplitude', 'Value'].values[0]
    print(sigmas_x*sigmas_y*amplitude)


if __name__ == "__main__":
   folder_path = r"T:\KAT1\Marijn\redmot\time of flight\nov15measurements\atom number"
   image_name = r"6_sf_final\0000.tif"

   main(folder_path, image_name)
