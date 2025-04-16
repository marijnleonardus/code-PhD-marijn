import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# append modules dir
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined modules
from image_analysis_class import ManipulateImage
from math_class import Math
from data_handling_class import reshape_roi_matrix


class RoiCounts:
    def __init__(self, center_weight, roi_radius):
        self.center_weight = center_weight
        self.roi_radius = roi_radius

    def weighted_count_roi(self, pixel_box):
        """compute weighted sum of counts in a pixel box using element wise multiplication 

        Args:
            weights_matrix (np 2d array): 
            pixel_box (np array): the ROI

        Returns:
            counts (int): weighted number of counts
        """
        
        weight_matrix = np.ones((2*self.roi_radius + 1, 2*self.roi_radius + 1))
        weight_matrix[self.roi_radius, self.roi_radius] = self.center_weight
        weighted_pixel_box = weight_matrix*pixel_box
        counts = np.sum(weighted_pixel_box)
        return counts
        
    def compute_pixel_sum_counts(self, images: list[np.ndarray], y_coords: np.ndarray, x_coords: np.ndarray):
            """
            Compute weighted pixel sums within Regions of Interest (ROIs) around specified coordinates
            across multiple images.

            Args:
                images (list[np.ndarray]): List of images (2D arrays) to process.
                y_coords (np.ndarray): Array of y-coordinates for ROI centers.
                x_coords (np.ndarray): Array of x-coordinates for ROI centers.

            Returns:
                tuple: 
                    - rois_matrix (np.ndarray): 4D array storing cropped ROIs (roi, image, px_y, px_x).
                    - rois_counts_matrix (np.ndarray): 2D array storing weighted ROI sums (roi, image).
            """
            # Initialize containers for storing ROIs and their weighted counts
            roi_crops = []  # List to store cropped ROIs for all ROIs across images
            roi_weighted_sums = []  # List to store weighted pixel sums for all ROIs across images

            num_rois = len(y_coords)  # Number of ROIs
            num_images = len(images)  # Number of images

            for roi_idx in range(num_rois):
                # Containers for current ROI across all images
                roi_crops_per_image = []
                weighted_sums_per_image = np.zeros(num_images)

                for image_idx, image in enumerate(images):
                    # Extract ROI as a cropped section of the image
                    cropped_roi = ManipulateImage().crop_array_center(
                        image, 
                        y_coords[roi_idx], 
                        x_coords[roi_idx], 
                        crop_radius=self.roi_radius
                    )
                    roi_crops_per_image.append(cropped_roi)

                    # Compute the weighted sum for the current ROI
                    weighted_sum = self.weighted_count_roi(cropped_roi)
                    weighted_sums_per_image[image_idx] = weighted_sum

                # Append data for this ROI to the main containers
                roi_crops.append(roi_crops_per_image)
                roi_weighted_sums.append(weighted_sums_per_image)

            # Convert results to numpy arrays
            rois_matrix = np.array(roi_crops)
            rois_counts_matrix = np.array(roi_weighted_sums)

            return rois_matrix, rois_counts_matrix
    
    def plot_average_of_roi(self, rois_list):
        """given a list of ROI pixel boxes, plot the average to check everything went correctly

        Args:
            rois_list (np array): list of ROIs
        """
        rois_array_3d = np.stack(rois_list, axis=0)
        average_image = np.mean(rois_array_3d, axis=0)
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(average_image)
        ax.set_title('Average pixel box for ROI 0')


class SingleAtoms:
    def calculate_histogram_detection_threshold(ampl0: float, mu0: float, sigma0: float, ampl1: float, mu1: float, sigma1: float):
        """calculate detection threshold for double gaussian fit
        found by settings g1(x) = g2(x) and solving for x (ABC formula)"""

        A = 1/(2*sigma0**2)-1/(2*sigma1**2)
        B = mu1/sigma1**2 - mu0/sigma0**2
        C = mu0**2/(2*sigma0**2) - mu1**2/(2*sigma1**2) + np.log(ampl1*sigma0/ampl0/sigma0)

        sols = Math.solve_quadratic_equation(A, B, C)
        # print("solutions", sols)
        
        # take solution between mu0 and mu1
        valid_sol = [x for x in [sols[0], sols[1]] if mu0 <= x <= mu1]
        valid_sol = np.round(valid_sol, 0)
        return valid_sol
    
    def calculate_survival_probability(images_path, binary_threshold):
        """
        Calculate the survival probability of atoms in ROIs based on ROI counts matrix and binary threshold.
        Images 0, 3, 5, etc. are initial images, and images 1, 2, 4, etc. are final images.
        Then calculates surv probability for each pair of images.
        
        Parameters:
        - roi_counts_matrix (numpy.ndarray): Matrix containing ROI counts for each image.
        - binary_threshold (int): Threshold for binary classification of ROI counts.
        
        Returns:
        - survival_probability (numpy.ndarray): Survival matrix indicating survival status of atoms in ROIs.
        """
        roi_counts_matrix = np.load(os.path.join(images_path, 'roi_counts_matrix.npy'))
        print("raw data: nr ROIs, nr images: ", np.shape(roi_counts_matrix))

        # Perform binary thresholding: entries above threshold become 1, others become 0
        binary_matrix = (roi_counts_matrix > binary_threshold).astype(int)

        # Number of image pairs: floor divide by 2
        num_pairs = binary_matrix.shape[1] // 2

        # Initialize survival matrix with NaNs (undefined by default)
        # Using a floating-point array allows us to use np.nan
        survival_matrix = np.full((binary_matrix.shape[0], num_pairs), np.nan, dtype=float)

        # Process each pair of images
        for im_idx in range(num_pairs):
            initial = binary_matrix[:, 2*im_idx]     
            final = binary_matrix[:, 2*im_idx + 1]    
            
            # Create a mask for ROIs that had an atom initially
            # For ROIs where there was an atom, 1 = atom survived, 0 = atom disappeared
            mask = (initial == 1)
            survival_matrix[mask, im_idx] = final[mask]

        # reshape roi_counts_matrix depending on the number of averages
        # laod x_values. If multiple averages used x values contains duplicates
        df = pd.read_csv(images_path + 'log.csv')
        x_values, survival_matrix_sorted = reshape_roi_matrix(df, survival_matrix)

        # compute average by summing over repeated values
        survival_probability = np.nanmean(survival_matrix_sorted, axis=2)
        return x_values, survival_probability
    