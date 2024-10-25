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
from decimal import Decimal

# append modules dir
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../../modules'))
sys.path.append(modules_dir)

# user defined functions
from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage
from image_analysis_class import SpotDetection

os.system('cls')


def main(folder_path, image_name):
    # Read the image using imageio and flatten to 1d
    image_original = CameraImage.load_image_from_file(folder_path, image_name)
    image_original_flat = image_original.ravel()

    SpotDetectionObject = SpotDetection(sigma=60, threshold_detection=0.08, image=image_original_flat)
    spots_laplaciangaussian = SpotDetectionObject.laplacian_of_gaussian_detection()
    print(spots_laplaciangaussian)

    # Initial guess for the parameters, bound theta parameter
    initial_guess = (100, 100, 500, 50, 50, 0, 200) # amplitude, x0, y0, sigma_x, sigma_y, theta, offset
    bounds = (0, [1000,2000, 2000, 300, 300, np.pi, 200])

   # Precompute meshgrid
    x_max, y_max = 0, 0
    x_max = max(x_max, image_original.shape[1])
    y_max = max(y_max, image_original.shape[0])
    x = np.arange(0, x_max, 1)
    y = np.arange(0, y_max, 1)
    xy_mesh = np.meshgrid(x,y)
    xy_flat = np.vstack((xy_mesh[0].ravel(), xy_mesh[1].ravel()))  # Flatten the meshgrid

    # Fit 2D Gaussian to the entire image with constrained theta
    params, _ = curve_fit(
        FittingFunctions.gaussian_2d_angled,
        xy_flat,
        image_original_flat,
        p0=initial_guess,
        bounds=bounds
    )    
    
    amplitude = params[0]
    x0 = params[1]
    y0 = params[2]
    sigma_x = params[3]
    sigma_y = params[4]
    angle = params[5]
    offset = params[6]

    sigx = 322e-6
    sigy = 204e-6
    crosssec = 3*(461e-9)**2/2/np.pi
    atom_nr = 2*np.pi*0.1*sigx*sigy/crosssec
    print("Atom number:", f"{atom_nr:.0e}")

    fig, ax = plt.subplots()
    ax.imshow(image_original)

    ellipse = Ellipse((x0, y0), width=sigma_x*2, edgecolor='r', facecolor='none', height=sigma_y*2, angle=angle)
    ax.add_patch(ellipse)
    plt.show()    

if __name__ == "__main__":
   folder_path = r"Z://Strontium//Images//2024-10-21//442187//"
   image_name = r"0000absorption.tif"
   main(folder_path, image_name)
