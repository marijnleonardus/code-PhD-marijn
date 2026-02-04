import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.integrate
from skimage.feature import blob_log
from scipy.stats import sem
import os

# user defined modules
from modules.math_class import Math
from modules.camera_image_class import CameraImage
from modules.fitting_functions_class import FittingFunctions
from utils.data_handling import sort_raw_measurements


class ROIs:
    def __init__(self, roi_radius: int, log_thresh: int):
        self.roi_radius = roi_radius
        self.patch_size = 2*roi_radius + 1
        self.log_thresh = log_thresh

    def _extract_rois(self, images: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray):
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
            rois[i] = padded[:, y:y + p, x:x + p]

        return rois

    def _plot_single_roi(self, avg_patches: np.ndarray):
        """(optional) quick sanity plot for single ROI, e.g. ROI #0"""
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        # internally, the counts are not multiplied, for the conversion to photons to work
        # but for visualization we multiply by patch_size^2, otherwise it is confusing to have 
        # a weight matrix that sums to 1
        im = ax.imshow(avg_patches[3]*self.patch_size**2, cmap='viridis') 
        #ax.set_title("Average patch used as filter for ROI #1")
        #ax.set_xlabel('px')
        #ax.set_ylabel('px')
        fig.colorbar(im, ax=ax)
        #plt.savefig("output/average_patch.png", dpi=300, bbox_inches='tight')

    def _plot_avg_stack(self, avg_stack: np.ndarray, detected_spots: list[np.ndarray]):
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(avg_stack, cmap='viridis')

        # annotate ROIs with circles
        y_coords, x_coords = detected_spots
        ax.scatter(x_coords, y_coords, s=30, facecolors='none', edgecolors='r', label='Detected ROIs')

        # annotate with ROI index
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            ax.text(
                x + 2, y + 2, # small offset so text doesn't sit on the marker
                str(i), color='white', fontsize=8, ha='left', va='bottom', bbox=dict(
                    facecolor='black', alpha=0.5, edgecolor='none', pad=1
                )
            )
        ax.set_title('Average image (z-projection)')
        ax.set_xlabel('x [px]')
        ax.set_ylabel('y [px]')
        fig.colorbar(im, ax=ax, label='Counts')
        ax.legend()

    def _sort_to_reading_order(self, spots, row_tolerance=10):
        """
        Sorts spots from top-left to bottom-right (reading order).
        It groups spots into rows based on Y-proximity (tolerance) to handle 
        slight grid rotation, then sorts each row by X.
        
        Args:
            spots: np.ndarray of shape (N, 3) -> (y, x, sigma)
            row_tolerance: float, pixels. Points within this Y-distance are considered same row.
        """
        if len(spots) == 0:
            return spots

        # Sort strictly by Y first to order them roughly top-to-bottom
        sorted_by_y = spots[spots[:, 0].argsort()]
        final_sorted_spots = []
        current_row = []
        
        # Initialize first row reference
        current_row_y_ref = sorted_by_y[0, 0]
        
        for spot in sorted_by_y:
            # Check if this spot belongs to the current row (is within tolerance of the row's Y)
            if abs(spot[0] - current_row_y_ref) < row_tolerance:
                current_row.append(spot)
            else:
                # Row finished: Sort the current row by X (column index) and add to final list
                current_row.sort(key=lambda k: k[1])
                final_sorted_spots.extend(current_row)
                # Start a new row
                current_row = [spot]
                current_row_y_ref = spot[0]
        # Don't forget to append and sort the very last row
        if current_row:
            current_row.sort(key=lambda k: k[1])
            final_sorted_spots.extend(current_row)
        return np.array(final_sorted_spots)

    def calculate_roi_counts(self, images_path: str, file_name_suffix: str, use_weighted_count: bool, roi_index_tolerance: int):
        """calculate ROI counts for each image in the stack

        Args:
            images_path (str): 
            file_name_suffix (str): 

        Raises:
            ValueError: if no images are loaded

        Returns:
            spots: np.ndarray of shape (N, 3) -> (y, x, sigma)
            roi_counts: ROI counts for each image in the stack, per ROI
        """
      
        # load your images
        image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)
        if image_stack.size == 0:
            raise ValueError("No images loaded, check path/suffix")
        print("loaded images:", image_stack.shape)
        
        # find detected spots and plot on top of average image
        mean_img = image_stack.mean(axis=0)
        spots = blob_log(mean_img, max_sigma=3, min_sigma=1, num_sigma=5, threshold=self.log_thresh)

        # sort spots from top-left to bottom-right (reading order)
        spots = self._sort_to_reading_order(spots, row_tolerance=roi_index_tolerance)

        y_coor, x_coor = spots[:, 0], spots[:, 1]
        detected_spots = [y_coor, x_coor]
        print(f"Detected {len(spots)} spots")
        self._plot_avg_stack(mean_img, detected_spots)

        # extract all patches into a 4D array of shape (n_rois, n_images, p, p)
        rois_mat = self._extract_rois(image_stack, y_coor, x_coor)

        if use_weighted_count:
            # er-ROI average patch, Shape (n_rois, p, p)
            avg_patches = rois_mat.mean(axis=1)                

            # Optional: clip negatives if you want purely positive templates
            avg_patches = avg_patches - avg_patches.min(axis=(1, 2), keepdims=True)

            # normalize so ∑_{m,n} templates[i,m,n] == 1
            sums = avg_patches.sum(axis=(1, 2), keepdims=True)
            templates = avg_patches/sums  # (n_rois, p, p)
        else:
            # plain average (= uniform) template also sums to 1 now
            templates = np.ones((rois_mat.shape[0], *rois_mat.shape[2:]))
            templates = templates/templates.sum(axis=(1,2), keepdims=True)

        # plot single ROI to check it went correctly
        self._plot_single_roi(templates)

        # apply matched‐filter on each patch
        # counts[i,j] = ∑_{m,n} rois_mat[i,j,m,n] * templates[i,m,n]
        roi_counts = np.einsum('ijnm,inm->ij', rois_mat, templates)

        # at the end multiply by nr of pixels, to get the number of counts in total ROI
        roi_counts *= (self.patch_size**2)
        return spots, roi_counts
    

class SingleAtoms():
    def __init__(self, binary_threshold: int, images_path: str):
        self.images_path = images_path
        self.binary_threshold = binary_threshold
    
    def _calculate_survival_probability_binary(self):
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
        roi_counts_matrix = np.load(os.path.join(self.images_path,'roi_counts_matrix.npy'))
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
    
    def reshape_survival_matrix(self, df: pd.DataFrame):
        """reshape survival matrix into shape (nr_rois, nr_x_values, nr_avgerages)

        Parameters:
        - df (pd dataframe): matrix containing raw measurements 
        
        Returns:
        - x_grid (np.ndarray): unique x values from the dataframe
        - survival_matrix (np.ndarray): survival probability (nr_rois, nr_x_values, nr_averages)
        """

        # sort the binary matrix based on x values, possibly number of x_values not equivalent to nr images
        # if exp stopped before
        survival_matrix_binary = self._calculate_survival_probability_binary()
        nr_avg, x_grid, survival_matrix_sorted = sort_raw_measurements(df, survival_matrix_binary)
    
        # Use the fixed sort_raw_measurements from the previous step
        # This will return survival_matrix_sorted with shape (nr_rois, n_actual_samples)
        _, x_grid, survival_matrix_sorted = sort_raw_measurements(df, survival_matrix_binary)

        nr_rois = survival_matrix_sorted.shape[0]
        n_samples = survival_matrix_sorted.shape[1]
        n_unique_x = len(x_grid)

        # Calculate the maximum number of FULL averages available for ALL x values
        # This ensures every x-value in the grid has the same number of shots
        nr_avg_complete = n_samples // n_unique_x

        # Truncate the sorted matrix to fit the perfect grid
        # We take only the first (n_unique_x * nr_avg_complete) columns
        valid_total_size = n_unique_x * nr_avg_complete
        survival_matrix_truncated = survival_matrix_sorted[:, :valid_total_size]

        # Reshape safely
        survival_matrix = survival_matrix_truncated.reshape(nr_rois, n_unique_x, nr_avg_complete)
        
        return x_grid, survival_matrix
    
    def calculate_survival_statistics(self, df: pd.DataFrame):
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


class BinaryThresholding():
    def __init__(self, popt: np.ndarray):
        """Initialize with fit parameters from double Gaussian fit"""
        self.ampl0 = popt[0]
        self.mu0 = popt[1]
        self.sigma0 = popt[2]
        self.ampl1 = popt[3]
        self.mu1 = popt[4]
        self.sigma1 = popt[5]

    def gauss_function_0peak(self, x):
        """Gaussian function with 0 peak, used for fitting"""
        
        return FittingFunctions.gaussian_function(x, 0, self.ampl0, self.mu0, self.sigma0)

    def gauss_function_1peak(self, x):
        """Gaussian function with 1 peak, used for fitting"""

        return FittingFunctions.gaussian_function(x, 0, self.ampl1, self.mu1, self.sigma1)

    def calculate_histogram_detection_threshold(self, filling_fraction):
        """calculate detection threshold for double gaussian fit found by 
        maximing the imaging fidelity by setting the derivative of the fidelity 
        to 0 and solving for x_t where x_t is the detection threshold."""

        A = 1/self.sigma1**2 - 1/self.sigma0**2
        B = 2*self.mu0/self.sigma0**2 - 2*self.mu1/self.sigma1**2
        C = self.mu1**2/self.sigma1**2 - self.mu0**2/self.sigma0**2 - 2*np.log(self.ampl1*filling_fraction/(self.ampl0*(1 - filling_fraction)))
        sols = Math.solve_quadratic_equation(A, B, C)
        
        # take solution between mu0 and mu1
        valid_sol = [x for x in [sols[0], sols[1]] if self.mu0 <= x <= self.mu1]
        valid_sol = np.round(valid_sol, 0)
        return valid_sol

    def calculate_imaging_fidelity(self, filling_fraction):
        """Calculate the imaging fidelity based on the area under the Gaussian 
        curves above the detection threshold.
        see Madjarov thesis, eq. 2.110"""

        threshold = self.calculate_histogram_detection_threshold(filling_fraction)

        # contribute contribution of 0 peak for total integrand
        contrib_peak0, _ = scipy.integrate.quad(self.gauss_function_0peak, -np.inf, threshold)
        total_peak0 = self.ampl0*self.sigma0*np.sqrt(2*np.pi)
        fidelity_0 = contrib_peak0/total_peak0

        # and for 1 peak 
        contrib_peak1, _ = scipy.integrate.quad(self.gauss_function_1peak, threshold, np.inf)
        total_peak1 = self.ampl1*self.sigma1*np.sqrt(2*np.pi)
        fidelity_1 = contrib_peak1/total_peak1

        fidelity = (1 - filling_fraction)*fidelity_0 + filling_fraction*fidelity_1
        return fidelity
    
