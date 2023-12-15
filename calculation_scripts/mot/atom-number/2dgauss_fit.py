# marijn venderbosch
# november 2023
"""script for analyzing time of flight data"""

# %% 
# standard libraries
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# append modules dir
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
print(modules_dir)
sys.path.append(modules_dir)

# user defined functions
from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage

 # %% functions

def fit_and_return_parameters(xy, data):
    """fit data and return parameters"""

    # Initial guess for the parameters
    initial_guess = (50, 300, 200, 20, 20, 0, np.min(data))
    
    # Define bounds for the parameters, including the constraint for theta
    bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, _ = curve_fit(FittingFunctions.gaussian_2d, xy, data, p0=initial_guess, bounds=bounds)

    # Create a DataFrame to store parameters and standard errors
    result = pd.DataFrame({
        'Parameter': ['Amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'Offset'],
        'Value': params
        })
    return result

def main(folder_path, image_name):
    # Read the image using imageio
    original_image = CameraImage.load_image_from_file(folder_path, image_name)

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

# %% execute script

if __name__ == "__main__":
   folder_path = r"T:\KAT1\Marijn\redmot\time of flight\nov15measurements\atom number"
   image_name = r"\6_sf_final\0000.tif"

   main(folder_path, image_name)
