# author: Marijn Venderbosch
# July 2024

import numpy as np
from numpy import unravel_index
from skimage.feature import blob_log
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# append modules dir
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)
from fitting_functions_class import FittingFunctions


class ManipulateImage:
    """to do; need to merge some of these cropping functions that are similar
    crop_array_edge and crop_array_center can be static methods i think"""
    def crop_array_edge(self, array, crop_range_x, crop_range_y):
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
    
    def crop_array_center(self, array, center_x, center_y, crop_radius):
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
    def crop_to_region_of_interest(image_file, roi_size):
        """crops image file to region of interest, used for MOT images"""
    
        # Finding center MOT
        location_maximum = image_file.argmax()
        indices = unravel_index(location_maximum, image_file.shape)
    
        # Crop image                                                     
        region_of_interest = image_file[indices[0] - roi_size:indices[0] + roi_size,
            indices[1] - roi_size:indices[1] + roi_size]
    
        # Normalize but keep max. number of counts to be used in atom number formula
        max_counts = np.max(region_of_interest)
        roi_normalized = region_of_interest/max_counts
        return roi_normalized, max_counts
    
    @staticmethod
    def crop_center(img, cropx, cropy):
        """crop image centered around middle"""
        
        y, x, *_ = img.shape
        startx = x//2 - (cropx//2)
        starty = y//2 - (cropy//2)    
        return img[starty:starty + cropy, startx:startx + cropx, ...]


class RoiCounts:
    def __init__(self, center_weight, roi_radius):
        self.center_weight = center_weight
        self.roi_radius = roi_radius

    def generate_weighted_counts(self):
        """generate a matrix of weights for weighted sum of counts in a pixel box, 
        for example 
        
        [1, 1, 1, 
        1, 3, 1, 
        1, 1, 1]
        
        for center weight 3
        """

        roi_weight_matrix = np.ones((2*self.roi_radius + 1, 2*self.roi_radius + 1))
        roi_weight_matrix[self.roi_radius, self.roi_radius] = self.center_weight
        return roi_weight_matrix

    def weighted_count_roi(self, pixel_box):
        """compute weighted sum of counts in a pixel box using element wise multiplication 

        Args:
            weights_matrix (np 2d array): 
            pixel_box (np array): the ROI

        Returns:
            counts (int): weighted number of counts
        """
        
        weights_matrix = self.generate_weighted_counts()
        weighted_pixel_box = weights_matrix*pixel_box
        counts = np.sum(weighted_pixel_box)
        return counts
    
    def compute_pixel_sum_counts(self, images_list, y_coor, x_coor):
        """define ROIs around the LOG spots, 
        and compute a weighted sum over the pixel count in the ROI"""

        rois_list = []*len(images_list)
        roi_counts_array = np.zeros(len(images_list))
        for im in range(len(images_list)):
            # define ROI as cropped image
            roi = ManipulateImage().crop_array_center(images_list[im], y_coor, x_coor, crop_radius=self.roi_radius)
            rois_list.append(roi)

            # get ROI counts (weighted)
            roi_weight_matrix = self.generate_weighted_counts()
            roi_count = self.weighted_count_roi(roi)
            roi_counts_array[im] = roi_count
        return rois_list, roi_counts_array
    
    def plot_average_of_roi(self, rois_list):
        """given a list of ROI pixel boxes, plot the average to check everything went correctly

        Args:
            rois_list (np array): list of ROIs
        """
        rois_array_3d = np.stack(rois_list, axis=0)
        average_image = np.mean(rois_array_3d, axis=0)
        plt.imshow(average_image)

class SpotDetectionFitting():
    def __init__(self, sigma, threshold_detection, image):
        
        self.sigma = sigma
        self.threshold_detection = threshold_detection
        self.image = image

    def laplacian_of_gaussian_detection(self):
        """do a gaussian blur, then compute laplacian. 
        if this is above a threhsold, return the location of this spot"""

        spots_laplacian_gaussian = blob_log(self.image, max_sigma=2*self.sigma, min_sigma=0.5*self.sigma,
            num_sigma=10, threshold=self.threshold_detection)
        return spots_laplacian_gaussian
    
    def twod_gaussian_fit(self, amplitude_guess, offset_guess, print_enabled, plot_enabled):
        """fit 2d gaussian

        Args:
            amplitude_guess (float): _description_
            offset_guess (float): _description_
            plot_enabled (float): _description_

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

    def total_pixel_count(self, window_radius, print_enabled, plot_enabled):
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
    