# author: Marijn Venderbosch
# July 2024

import numpy as np
from skimage.feature import blob_log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.transforms import ScaledTranslation
import os
import sys

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined modules
from fitting_functions_class import FittingFunctions
from camera_image_class import CameraImage
from units import mm


class ManipulateImage:
    """to do; need to merge some of these cropping functions that are similar
    crop_array_edge and crop_array_center can be static methods i think"""
    def crop_array_edge(self, array: np.ndarray, crop_range_x: int, crop_range_y: int):
        """
        Crops an array by removing `x` number of rows and columns from each side.

        Parameters:
            array (numpy.ndarray): The input array to be cropped.
            crop_range_x (int): Nr of columns to be cropped from each side
            crop_range_y (int): Nr of rows to be cropped from each side

        Returns:
            cropped_array (numpy.ndarray): The cropped array. 
        """
  
        cropped_array = array[crop_range_y:-crop_range_y, crop_range_x:-crop_range_x]
        return cropped_array
    
    def crop_array_center(self, array: np.ndarray, center_x: int, center_y: int, crop_radius: int):
        """crop ROI from center position (column, row)

        Args:
            array (numpy.ndarray): input array
            center_x (int): center of image (column)
            center_y (int): centerof image (row)
            crop_radius (int): amount of pixels to keep in each direction. 

        Returns:
            center_roi (numpy.ndarray): _description_
        """

        rows_start = int(center_y - crop_radius)
        rows_end = int(center_y + (crop_radius) + 1)
        cols_start = int(center_x - (crop_radius))
        cols_end = int(center_x + crop_radius + 1)
        center_roi = array[cols_start:cols_end, rows_start:rows_end]
        return center_roi
    
    @staticmethod
    def crop_to_region_of_interest(image_file: np.ndarray, row: int, column: int, roi_size: int):
        """crops image file to region of interest defined by (column, row), used for MOT images"""
    
        # cropping indices
        left_column = int(column - roi_size/2)
        right_column = int(column + roi_size/2)
        bottom_row = int(row - roi_size/2)
        top_row = int(row + roi_size/2)
    
        # Crop image                                                     
        region_of_interest = image_file[bottom_row:top_row, left_column:right_column]
        return region_of_interest
    
    @staticmethod
    def crop_center(img: np.ndarray, crop_x: int, crop_y: int):
        """crop image centered around middle"""
        
        y, x, *_ = img.shape
        startx = x//2 - (crop_x//2)
        starty = y//2 - (crop_y//2)    
        return img[starty:starty + crop_y, startx:startx + crop_x, ...]


class SpotDetectionFitting():
    def __init__(self, sigma: float, threshold_detection: float, image: np.ndarray):
        self.sigma = sigma
        self.threshold_detection = threshold_detection
        self.image = image

    def laplacian_of_gaussian_detection(self):
        """do a gaussian blur, then compute laplacian. 
        if this is above a threhsold, return the location of this spot"""

        spots_laplacian_gaussian = blob_log(self.image, max_sigma=2*self.sigma, min_sigma=0.5*self.sigma,
            num_sigma=10, threshold=self.threshold_detection)
        return spots_laplacian_gaussian
    
    def twod_gaussian_fit(self, amplitude_guess: float, offset_guess: float, print_enabled: bool, plot_enabled: bool):
        """fit 2d gaussian

        Args:
            amplitude_guess (float): _description_
            offset_guess (float): _description_
            print_enabled (Boolean): 
            plot_enabled (Boolean): 

        Returns:
            amplitude, sigma_x, sigma_y (list): fit parameters of 2d gaussian result
        """

        # Read the image using imageio and flatten to 1d
        image_flat = self.image.ravel()

        # detect spots using laplacian of gaussian filter
        #SpotDetectionObject = SpotDetection(sigma=expected_pixel_size, threshold_detection=0.06, image=image_original)
        spots_laplaciangaussian = self.laplacian_of_gaussian_detection()
        if print_enabled:
            print("detected spot: row, column, radius", spots_laplaciangaussian)
        blob_y, blob_x, blob_radius = spots_laplaciangaussian[0]

        # Initial guess for the parameters, bound theta parameter
         # amplitude, x0, y0, sigma_x, sigma_y, theta, offset
        initial_guess = (amplitude_guess, blob_x, blob_y, blob_radius*.5, blob_radius*.5, np.pi/4, offset_guess)
        bounds = (0, [200,2000, 2000, 300,  300, np.pi, 300])

        # Define mesh grid for fitting, flatten for fitting
        x_max, y_max = 0, 0
        x_max = max(x_max, self.image.shape[1])
        y_max = max(y_max, self.image.shape[0])
        x = np.arange(0, x_max, 1)
        y = np.arange(0, y_max, 1)
        xy_mesh = np.meshgrid(x, y)
        xy_flat = np.vstack((xy_mesh[0].ravel(), xy_mesh[1].ravel())) 

        # Fit 2D Gaussian to the entire image
        fit_params, _ = curve_fit(
            FittingFunctions.gaussian_2d_angled,
            xy_flat,
            image_flat,
            p0=initial_guess,
            bounds=bounds
        )
        amplitude, x0, y0, sigma_x, sigma_y, rotation_angle, offset = fit_params
        if print_enabled:
            print("fitted amplitude: ", amplitude) 
            print("fitted peak location (x,y): ", x0, y0)
            print("fitted angle: ", rotation_angle*180/np.pi, " degrees")
            print("fitted offset: ", offset)

        if plot_enabled:
            fig, ax = plt.subplots()
            ax.imshow(self.image, cmap="jet")
            ellipse = Ellipse((x0, y0), width=sigma_x*2, edgecolor='r', angle=-rotation_angle*180/np.pi, 
                facecolor='none', height=sigma_y*2)
            cross = Ellipse((blob_x, blob_y), width=blob_radius, edgecolor='b',
                facecolor='none', height=blob_radius)
            ax.add_patch(ellipse)
            ax.add_patch(cross)
            plt.show()    

        return amplitude, sigma_x, sigma_y

    def total_pixel_count(self, window_radius: int, print_enabled: bool, plot_enabled: bool):
        """compute total pixel count for absorption image

        Args:
            window_radius (int): radius of window around image to crop around
            print_enabled (Boolean): 
            plot_enabled (Boolean): 
        """
        spots_laplaciangaussian = self.laplacian_of_gaussian_detection()
        blob_y, blob_x, _ = spots_laplaciangaussian[0]
        if print_enabled:
            print(spots_laplaciangaussian)

        Manipulate = ManipulateImage()
        array_cropped = Manipulate.crop_array_center(array=self.image, center_x=blob_y,
            center_y=blob_x, crop_radius=window_radius)

        if plot_enabled:
            fig, ax = plt.subplots()
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            im1 = ax.imshow(array_cropped, cmap="jet")
            fig.colorbar(im1, cax=cbar_ax)
            plt.show()

        # we use an offset of 200 px counts to prevent negative values. 
        offset = 200
        rows = np.shape(array_cropped)[0]
        cols = np.shape(array_cropped)[1]
        total_pixel_count = np.sum(array_cropped) - rows*cols*offset
        return total_pixel_count


class ImageStats():
    @staticmethod
    def calculate_uniformity(input_vector: np.ndarray):
        """calculate uniformity in intensities for example
        from https://doi.org/10.1364/OE.15.001913

        Args:
            input_vector (np array): 1d array of pixel values
        Returns:
            uniformity (float): uniformity value between 0 and 1. 1 = perfect uniformity
        """    

        uniformity = 1 - (max(input_vector) - min(input_vector))/(max(input_vector) + min(input_vector))
        return uniformity


class MOTPlot():
    def __init__(self, image: np.ndarray, magnification: float, px_size: float, binning: int):
        self.magnification = magnification
        self.px_size = px_size
        self.binning = binning
        self.image = image
        
    def plot_with_scaled_axes(self):
        pixels_y = self.image.shape[0]
        pixels_x = self.image.shape[1]
        roi_size_y = CameraImage.pixels_to_m(pixels_y, self.magnification, self.px_size, self.binning)
        roi_size_x = CameraImage.pixels_to_m(pixels_x, self.magnification, self.px_size, self.binning)
        
        figwidth = 0.5*3.375  # inches
        figheight = (3.375*0.5)*0.61
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
        ax.set_xlabel(r'x [mm]')
        ax.set_ylabel(r'y [mm]')
        im = ax.imshow(self.image, cmap="jet", extent=[0, roi_size_x/mm, 0, roi_size_y/mm])
        fig.colorbar(im, ax=ax, label='Optical Density')

    def plot_withscalebar(self, scalebar_length_mm: float):
        pixels_y = self.image.shape[0]
        pixels_x = self.image.shape[1]
        roi_size_y = CameraImage.pixels_to_m(pixels_y, self.magnification, self.px_size, self.binning)
        roi_size_x = CameraImage.pixels_to_m(pixels_x, self.magnification, self.px_size, self.binning)

        figwidth = 0.7  # inches
        figheight = figwidth/0.61
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
        im = ax.imshow(self.image, cmap="Blues", extent=[0, roi_size_x/mm, 0, roi_size_y/mm])
        #fig.colorbar(im, ax=ax, label='Counts')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r'$\mathrm{ToF}=10\,\mathrm{ms}$')
        ## add coordinate axes
        origin_fraction = (0.1, 0.1) # Where the dot sits (bottom-left)
        pixel_length = 10              # Length of arrows in points
        arrow_props = dict(arrowstyle='-|>', lw=1.0, color='black', mutation_scale=5)
        trans_x = ax.transAxes + ScaledTranslation(pixel_length/72, 0, fig.dpi_scale_trans)
        trans_y = ax.transAxes + ScaledTranslation(0, pixel_length/72, fig.dpi_scale_trans)
        ax.plot(*origin_fraction, 'ko', markersize=1, transform=ax.transAxes, clip_on=False)
        ax.annotate('', 
            xycoords=trans_x, xy=origin_fraction,      # Tip (moved by pixel_length)
            textcoords=ax.transAxes, xytext=origin_fraction, # Tail (at origin)
            arrowprops=arrow_props)
        ax.annotate('', 
            xycoords=trans_y, xy=origin_fraction,      # Tip (moved by pixel_length)
            textcoords=ax.transAxes, xytext=origin_fraction, # Tail (at origin)
            arrowprops=arrow_props)
        ax.text(*origin_fraction, ' x', transform=trans_x, va='center', ha='left')
        ax.text(*origin_fraction, ' y', transform=trans_y, va='bottom', ha='center')

        # add scale bar
        scalebar_length_m = scalebar_length_mm*mm
        scalebar_x_start = 0.2*roi_size_x/mm
        scalebar_y_start = 0.9*roi_size_y/mm
        ax.hlines(y=scalebar_y_start, xmin=scalebar_x_start,
            xmax=scalebar_x_start + scalebar_length_m/mm, colors='black', linewidth=3)
        ax.text(scalebar_x_start + scalebar_length_m/(2*mm), scalebar_y_start - 0.02*(roi_size_y/mm),
            f'{scalebar_length_mm} mm', ha='center', va='top')
