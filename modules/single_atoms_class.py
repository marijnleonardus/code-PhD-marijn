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
from math_class import Math
from data_handling_class import sort_raw_measurements, compute_error_bernouilli
from typing import Union
from camera_image_class import CameraImage
from skimage.feature import blob_log
 


class ROIs:
    def __init__(self, roi_radius, center_weight):
        self.roi_radius = roi_radius
        size = 2*roi_radius + 1
        w = np.ones((size, size), dtype=np.float32)
        w[roi_radius, roi_radius] = center_weight
        self.weight_matrix = w

    def compute_pixel_sum_counts(self, images: Union[list[np.ndarray], np.ndarray], y_coords: np.ndarray, x_coords: np.ndarray) -> tuple: 
        """
        Compute weighted ROI counts, vectorized over images.
        apply padding to avoid out of index errors

        Args:
            images: list of 2D arrays or a 3D array of shape (n_images, H, W).
            y_coords, x_coords: float arrays of length n_rois.
        """
        # stack into array
        image_stack = np.asarray(images)
        n_images, H, W = image_stack.shape

        # round float coords to ints
        y_idx = np.rint(y_coords).astype(int)
        x_idx = np.rint(x_coords).astype(int)

        r = self.roi_radius
        p = 2*r + 1

        # single pad for all images
        padded = np.pad(image_stack, ((0,0), (r, r), (r, r)), mode='constant', constant_values=0)

        n_rois = len(y_idx)
        rois_matrix = np.empty((n_rois, n_images, p, p), dtype=image_stack.dtype)
        counts = np.empty((n_rois, n_images), dtype=np.float32)

        for i, (y, x) in enumerate(zip(y_idx, x_idx)):
            patch_cube = padded[:, y : y+p, x : x+p]
            rois_matrix[i] = patch_cube
            counts[i] = (patch_cube * self.weight_matrix).sum(axis = (1, 2))
        return rois_matrix, counts
    
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

    def calculate_roi_counts(self, images_path, file_name_suffix):
        """calculate ROI counts for all images in a directory
        and return the result"""

        # variables
        rois_radius = 1  # ROI size. Radius 1 means 3x3 array
        log_threshold = 10 # laplacian of gaussian kernel sensitivity
        weight_center_pixel = 1

        # images without cropping ('raw' data)
        image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)
        images_list = [image_stack[i] for i in range(image_stack.shape[0])]

        if np.shape(image_stack)[0] == 0:
            raise ValueError("No images loaded, check image path and file name suffix")
        else:
            print("nr images, pixels, pixels", np.shape(image_stack))

        # detect laplacian of gaussian spot locations from avg. over all images
        z_project = np.mean(image_stack, axis=0)
        spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=3, threshold=log_threshold)
        y_coor = spots_LoG[:, 0] 
        x_coor = spots_LoG[:, 1]
        print(spots_LoG)
        print("nr spots detected", np.shape(spots_LoG)[0])

        # plot average image and mark detected maximum locations in red, check if LoG was correctly detected
        fig1, ax1 = plt.subplots()
        ax1.imshow(z_project, cmap='gist_yarg')
        ax1.scatter(x_coor, y_coor, marker='x', color='r')
        fig1.show()
        ax1.set_title('Average image and LoG detected spots')

        # compute nr of counts in each ROI 
        ROIcounts = ROIs(rois_radius, weight_center_pixel)
        image_stack = np.stack(images_list, axis=0)    # shape: (n_images, H, W)
        rois_matrix, roi_counts_matrix = self.compute_pixel_sum_counts(
            image_stack, y_coor, x_coor
        )
        # plot average pixel box for ROI 1 to check everything went correctly
        ROIcounts.plot_average_of_roi(rois_matrix[0, :, :, :])
        plt.show()

        # (nr_rois, nr_images)
        return roi_counts_matrix

    @staticmethod
    def calculate_histogram_detection_threshold(fit_params: np.ndarray):
        """calculate detection threshold for double gaussian fit
        found by settings g1(x) = g2(x) and solving for x (ABC formula)"""

        # obtain fit parameters
        ampl0 = fit_params[0]
        mu0 = fit_params[1]
        sigma0 = fit_params[2]
        ampl1 = fit_params[3]
        mu1 = fit_params[4]
        sigma1 = fit_params[5]

        A = 1/(2*sigma0**2)-1/(2*sigma1**2)
        B = mu1/sigma1**2 - mu0/sigma0**2
        C = mu0**2/(2*sigma0**2) - mu1**2/(2*sigma1**2) + np.log(ampl1*sigma0/ampl0/sigma0)

        sols = Math.solve_quadratic_equation(A, B, C)
        # print("solutions", sols)
        
        # take solution between mu0 and mu1
        valid_sol = [x for x in [sols[0], sols[1]] if mu0 <= x <= mu1]
        valid_sol = np.round(valid_sol, 0)
        return valid_sol
    

class SingleAtoms():
    def __init__(self, binary_threshold, images_path):
        self.images_path = images_path
        self.binary_threshold = binary_threshold
    
    def calculate_survival_probability_binary(self):
        """
        Calculate the survival probability of atoms in ROIs based on ROI counts matrix and binary threshold.
        Images 0, 3, 5, etc. are initial images, and images 1, 2, 4, etc. are final images.
        Then calculates surv probability for each pair of images.
        
        Parameters:
        - roi_counts_matrix (numpy.ndarray): Matrix containing ROI counts for each image.
        - binary_threshold (int): Threshold for binary classification of ROI counts.
        
        Returns:
        - survival_matrix_binary (numpy.ndarray): 
            Survival matrix indicating survival status of atoms in ROIs as 0 or 1
        """
        print(self.images_path)
        roi_counts_matrix = np.load(os.path.join(self.images_path, 'roi_counts_matrix.npy'))
        print("raw data: nr ROIs, nr images: ", np.shape(roi_counts_matrix))

        # Perform binary thresholding: entries above threshold become 1, others become 0
        binary_matrix = (roi_counts_matrix > self.binary_threshold).astype(int)

        # Number of image pairs: floor divide by 2
        num_pairs = binary_matrix.shape[1]//2

        # Initialize survival matrix with NaNs (undefined by default)
        # Using a floating-point array allows us to use np.nan
        survival_matrix_binary = np.full((binary_matrix.shape[0], num_pairs), np.nan, dtype=float)

        # Process each pair of images
        for im_idx in range(num_pairs):
            initial = binary_matrix[:, 2*im_idx]     
            final = binary_matrix[:, 2*im_idx + 1]    
            
            # Create a mask for ROIs that had an atom initially
            # For ROIs where there was an atom, 1 = atom survived, 0 = atom disappeared
            mask = (initial == 1)
            survival_matrix_binary[mask, im_idx] = final[mask]
        return survival_matrix_binary
    
    def calculate_avg_sem_survival(self, df):
        """calculate avg and error of survival probability for each x value
        based on the binary survival matrix
        """
        # sort the binary matrix based on x values
        survival_matrix_binary = self.calculate_survival_probability_binary()
        nr_avg, x_values, survival_matrix_sorted = sort_raw_measurements(df, survival_matrix_binary)
    
        # Reshape now that duplicates are consecutive
        nr_rois = np.shape(survival_matrix_binary)[0]
        survival_matrix = survival_matrix_sorted.reshape(nr_rois, len(x_values), nr_avg)

        # compute average by summing over repeated values
        surv_prob = np.nanmean(survival_matrix, axis=2)

        # compute error
        sem_surv_prob = compute_error_bernouilli(nr_avg, surv_prob)
        return x_values, surv_prob, sem_surv_prob


def main():
    # variables
    images_path = 'Z:\\Strontium\\Images\\2025-04-17\\scan131340\\'  # path to images
    file_name_suffix = 'image'  # import files ending with image.tif
    weight_center_pixel = 3
    roi_radius = 1  # ROI size. Radius 1 means 3x3 array
    ROIobject = ROIs(roi_radius, weight_center_pixel)
    ROIobject.calculate_roi_counts(images_path, file_name_suffix)


if __name__ == "__main__":
    main()
