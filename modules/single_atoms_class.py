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
from data_handling_class import sort_raw_measurements
from plotting_class import Plotting
from camera_image_class import CameraImage
from skimage.feature import blob_log
from scipy.stats import sem



class ROIs:
    def __init__(self, roi_radius):
        self.roi_radius = roi_radius
        self.patch_size = 2*roi_radius + 1

    def extract_rois(self, images: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray):
        """
        Slice out p×p patches around each (y,x) from every image.
        Returns:
            rois_matrix: shape (n_rois, n_images, p, p)
        """
        r = self.roi_radius
        padded = np.pad(images, ((0,0), (r, r), (r, r)), mode='constant', constant_values=0)
        n_rois = len(y_coords)
        n_images = images.shape[0]
        p = self.patch_size

        rois = np.empty((n_rois, n_images, p, p), dtype=images.dtype)
        y_idx = np.rint(y_coords).astype(int)
        x_idx = np.rint(x_coords).astype(int)

        for i, (y, x) in enumerate(zip(y_idx, x_idx)):
            rois[i] = padded[:, y:y+p, x:x+p]

        return rois

    def plot_single_roi(self, avg_patches):
        """(optional) quick sanity plot for single ROI, e.g. ROI #0"""
        fig, ax = plt.subplots()
        im = ax.imshow(avg_patches[3], cmap='viridis')
        ax.set_title("Average patch used as filter for ROI #1")
        fig.colorbar(im, ax=ax)
        ax.axis('off')

    def calculate_roi_counts(self, images_path, file_name_suffix, use_weighted_count):
        """calculate ROI counts for each image in the stack

        Args:
            images_path (str): 
            file_name_suffix (str): 

        Raises:
            ValueError: if no images are loaded

        Returns:
            roi_counts: ROI counts for each image in the stack, per ROI
        """
        background_px_count = 500

        # 1) load your images
        image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)
        if image_stack.size == 0:
            raise ValueError("No images loaded, check path/suffix")
        print("loaded images:", image_stack.shape)

        # 2) find your spot centers via LoG on the mean image
        mean_img = image_stack.mean(axis=0)
        spots = blob_log(mean_img, max_sigma=3, min_sigma=1, num_sigma=5, threshold=12)
        y_coor, x_coor = spots[:,0], spots[:,1]
        print(f"Detected {len(spots)} spots")

        # 3) extract all patches into a 4D array of shape (n_rois, n_images, p, p)
        rois_mat = self.extract_rois(image_stack, y_coor, x_coor)

        if use_weighted_count:
            # 4) per-ROI average patch, background subtracted
            avg_patches = rois_mat.mean(axis=1)                 # (n_rois, p, p)
            #avg_patches = avg_patches - background_px_count

            # Optional: clip negatives if you want purely positive templates
            avg_patches = avg_patches - avg_patches.min(axis=(1,2), keepdims=True)

            # 5) normalize so ∑_{m,n} templates[i,m,n] == 1
            sums        = avg_patches.sum(axis=(1,2), keepdims=True)
            templates   = avg_patches / sums                    # (n_rois, p, p)
        else:
            # plain average (= uniform) template also sums to 1 now
            templates = np.ones((rois_mat.shape[0], *rois_mat.shape[2:]))
            templates = templates / templates.sum(axis=(1,2), keepdims=True)

        # plot single ROI to check it went correctly
        self.plot_single_roi(templates)

        # 6) apply matched‐filter on each patch
        #    counts[i,j] = ∑_{m,n} rois_mat[i,j,m,n] * templates[i,m,n]
        roi_counts = np.einsum('ijnm,inm->ij', rois_mat, templates)

        # at the end multiply by nr of pixels, to get the number of counts in total ROI
        roi_counts *= (self.patch_size**2)

        return roi_counts

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
        roi_counts_matrix = np.load(self.images_path + 'roi_counts_matrix.npy')
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
    
    def reshape_survival_matrix(self, df):
        """reshape survival matrix into shape (nr_rois, nr_x_values, nr_avgerages)

        Parameters:
        - df (pd dataframe): matrix containing raw measurements 
        
        Returns:
        - x_grid (np.ndarray): unique x values from the dataframe
        - survival_matrix (np.ndarray): survival probability (nr_rois, nr_x_values, nr_averages)
        """

        # sort the binary matrix based on x values
        survival_matrix_binary = self.calculate_survival_probability_binary()
        nr_avg, x_grid, survival_matrix_sorted = sort_raw_measurements(df, survival_matrix_binary)
    
        # Reshape now that duplicates are consecutive
        nr_rois = np.shape(survival_matrix_binary)[0]
        survival_matrix = survival_matrix_sorted.reshape(nr_rois, len(x_grid), nr_avg)
        return x_grid, np.array(survival_matrix)
    
    def calculate_survival_statistics(self, df):
        """from the survival probabilty matrix in form (nr_rois, nr_x_values, nr_averages)
        compute mean and standard error both per ROI and globally

        Returns:
            surv_prob_per_roi (np.ndarray): 
            surv_prob_global (np.ndarray):
            surv_prob_per_roi_sem (np.ndarray):
            surv_prob_global_sem (np.ndarray):
        """

        _, survival_matrix = self.reshape_survival_matrix(df)

        # calculate avg survival per ROI
        surv_prob_per_roi = np.nanmean(survival_matrix, axis=2)
        sem_per_roi = sem(survival_matrix, axis=2, nan_policy='omit')

        # calculate global surv prob.
        surv_prob_global = np.nanmean(surv_prob_per_roi, axis=0)

        # calculate sem globally
        # To calculate the global standard error correctly, we should first
        # calculate the average survival for each 'nr_x_value' across all ROIs for each run.
        # Then, calculate the SEM of these run averages.
        surv_prob_global_per_run = np.nanmean(survival_matrix, axis=0) # shape (nr_x_values, nr_avg)

        # Calculate the SEM along the 'nr_avg' axis (axis=1 for the new matrix)
        global_sem = sem(surv_prob_global_per_run, axis=1) # Shape: (nr_x_values,)

        stats_matrix = surv_prob_per_roi, surv_prob_global, sem_per_roi, global_sem
        return stats_matrix


def main():
    # variables
    images_path = 'Z:\\Strontium\\Images\\2025-04-17\\scan131340\\'  # path to images
    file_name_suffix = 'image'  # import files ending with image.tif
    roi_radius = 2  # ROI size. Radius 1 means 3x3 array
    ROIobject = ROIs(roi_radius)
    ROIobject.calculate_roi_counts(images_path, file_name_suffix)
    Plotting.savefig("output", "average_patch.png")


if __name__ == "__main__":
    main()
