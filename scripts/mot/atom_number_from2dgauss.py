# marijn venderbosch
# november 2023
"""script for analyzing time of flight data"""

# standard libraries
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# append modules dir
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
print(modules_dir)
sys.path.append(modules_dir)

# user defined functions
from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage
from image_analysis_class import SpotDetection


def fit_and_return_parameters(xy, data):
    """fit data and return parameters"""

    # Initial guess for the parameters, bound theta parameter
    initial_guess = (50, 300, 200, 20, 20, 0, np.min(data))
    bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.inf])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, _ = curve_fit(FittingFunctions.gaussian_2d_angled, xy, data, p0=initial_guess, bounds=bounds)

    # Create a DataFrame to store parameters and standard errors
    
    return result


def main(folder_path, image_name):
    # Read the image using imageio and flatten to 1d
    image_original = CameraImage.load_image_from_file(folder_path, image_name)
    image_flattened = image_original.flatten()

    SpotDetectionObject = SpotDetection(sigma=60, threshold_detection=0.0925, image=data)
    spots_laplaciangaussian = SpotDetectionObject.laplacian_of_gaussian_detection()
    print(spots_laplaciangaussian)

    # Initial guess for the parameters, bound theta parameter
    initial_guess = (100, 100, 500, 50, 50, 0, 200) # amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    bounds = (0, [1000,2000, 2000, 300, 300, np.pi, 200])

    # Fit 2D Gaussian to the entire image with constrained theta
    params, _ = curve_fit(FittingFunctions.gaussian_2d_angled, xy, data, p0=initial_guess, bounds=bounds)

    result = pd.DataFrame({
        'Parameter': ['Amplitude', 'xo', 'yo', 'sigma_x', 'sigma_y', 'theta', 'Offset'],
        'Value': params
        })

    # Precompute meshgrid
    x_max, y_max = 0, 0
    x_max = max(x_max, image_original.shape[1])
    y_max = max(y_max, image_original.shape[0])
    x = np.arange(0, x_max, 1)
    y = np.arange(0, y_max, 1)

    # empty lists to fill later
    x, y = np.meshgrid(x, y)        

    sigma_x = result.loc[result['Parameter'] == 'sigma_x', 'Value'].values[0]
    print("sigma_x: ", sigma_x)
    sigma_y = result.loc[result['Parameter'] == 'sigma_y', 'Value'].values[0]
    print("sigmas_y: ", sigma_y)
    amplitude = result.loc[result['Parameter'] == 'Amplitude', 'Value'].values[0]
    print("amplitude: ", amplitude)
    x0 = result.loc[result['Parameter'] == 'xo', 'Value'].values[0]
    print(x0)
    y0 = result.loc[result['Parameter'] == 'yo', 'Value'].values[0]
    print(y0)
    angle = result.loc[result['Parameter'] == 'theta', 'Value'].values[0]
    print("angle", angle)

    sigx = 322e-6
    sigy = 204e-6
    crosssec = 3*(461e-9)**2/2/np.pi
    atom_nr = 2*np.pi*0.1*sigx*sigy/crosssec
    print(atom_nr/1e5)

    fig, ax = plt.subplots()
    ax.imshow(image_original)

    ellipse = Ellipse((x0, y0), width=sigma_x*2, edgecolor='r', facecolor='none', height=sigma_y*2, angle=angle)
    ax.add_patch(ellipse)
    plt.show()    

folder_path = r"Z://Strontium//Images//2024-10-21//442187//"
image_name = r"0000absorption.tif"
main(folder_path, image_name)
