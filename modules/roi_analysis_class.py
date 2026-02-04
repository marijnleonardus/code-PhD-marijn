"""
Author: Marijn Venderbosch
Date: 2023-2026

This script contains classes to handle data analysis of raw tweezer images,
such that in the analysis scripts, we can do 

'SurvivalAnalysis.get_survival_data(**settings**)'

The order of the class methods is from low level (raw data) to high level (processed data)
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
from pathlib import Path
import shutil
import scipy.integrate
from skimage.feature import blob_log
from scipy.stats import sem
import os

# user defined modules
from modules.math_class import Math
from modules.camera_image_class import CameraImage
from modules.fitting_functions_class import FittingFunctions
from modules.camera_image_class import EMCCD
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
    

class SingleAtoms:
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


class BinaryThresholding:
    def __init__(self, popt: np.ndarray):
        """Initialize with fit parameters from double Gaussian fit"""
        self.popt = popt
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
    
    def calculate_fidelity_error(self, pcov: np.ndarray, filling_fraction: float, n_samples: int=10000):
        """
        Performs Monte Carlo simulation to estimate standard error of the fidelity.
        
        Args:
            pcov (np.ndarray): Covariance matrix of the fit parameters (6x6).
            filling_fraction (float): The atomic filling fraction.
            n_samples (int): Number of MC iterations.
            
        Returns:
            tuple: (mean_fidelity, std_err_fidelity)
        """
        print(f"Running Monte Carlo error analysis on {n_samples} samples...")
        
        # Generate random parameter sets based on fit covariance
        # The 'mean' is the current best-fit parameters (self.popt)
        samples = multivariate_normal.rvs(mean=self.popt, cov=pcov, size=n_samples)
        fidelities = []
        for params in samples:
            # params = [ampl0, mu0, sigma0, ampl1, mu1, sigma1]
            # Check physical constraints (Sigmas and Amplitudes must be positive)
            if params[0] <= 0 or params[2] <= 0 or params[3] <= 0 or params[5] <= 0:
                continue

            # Create a temporary instance for this sample
            # self.__class__ allows creating a new instance of BinaryThresholding dynamically
            temp_model = self.__class__(params)
            
            # Calculate fidelity for this sample
            # The method internally calculates the new threshold for these specific params
            fid = temp_model.calculate_imaging_fidelity(filling_fraction)
            
            if fid > 0: # Ensure valid result
                fidelities.append(fid)
        
        fidelities = np.array(fidelities)
        
        if len(fidelities) == 0:
            print("Warning: No valid Monte Carlo samples found.")
            return 0.0, 0.0
            
        mean_fid = np.mean(fidelities)
        std_err = np.std(fidelities) # Standard deviation of the distribution of fidelities
        
        return mean_fid, std_err


class ROICounts:
    """
    Orchestrates the analysis of ROI counts from raw images.
    Integrates ROI extraction, histogram fitting, threshold calculation, and data saving.
    """
    def __init__(self, roi_geometry: tuple[int, int], roi_params: dict, hist_params: dict, output_root: str):
        """
        Args:
            roi_geometry: (nr_rows, nr_cols)
            roi_params: dict with keys 'radius', 'log_thresh', 'index_tolerance'
            hist_params: dict with keys 'nr_bins_roi', 'nr_bins_avg', 'plot_only_initial'
            output_root: Path string for output storage
        """
        self.nr_rows, self.nr_cols = roi_geometry
        self.roi_radius = roi_params.get('radius', 1)
        self.log_thresh = roi_params.get('log_thresh', 10)
        self.roi_idx_tolerance = roi_params.get('index_tolerance', 4)
        
        self.nr_bins_roi = hist_params.get('nr_bins_roi', 12)
        self.nr_bins_avg = hist_params.get('nr_bins_avg', 50)
        self.plot_only_initial = hist_params.get('plot_only_initial', True)
        
        self.output_root = Path(output_root)

    def process_dataset(self, images_path: str, rid: str, file_suffix: str = 'image', show_photon_hist: bool = False):
        """
        Main driver method to process a specific scan (rid).
        """

        # Setup paths
        full_import_path = Path(images_path) / rid
        output_dir = self.output_root / 'roi_counts' / rid
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- Processing {rid} ---")

        # Calculate Counts
        rois_obj = ROIs(self.roi_radius, self.log_thresh)
        # Note: Converting Path to string because imports usually expect str
        _, roi_counts_matrix = rois_obj.calculate_roi_counts(
            str(full_import_path) + os.sep, 
            file_suffix, 
            use_weighted_count=True, 
            roi_index_tolerance=self.roi_idx_tolerance
        )
        
        print("Raw data shape (nr ROIs, nr shots): ", np.shape(roi_counts_matrix))
        np.save(output_dir / 'roi_counts_matrix.npy', roi_counts_matrix)

        # Prepare Data for Histogram (Filter Initial/Survival images)
        if self.plot_only_initial:
            # Slicing: take every 2nd image starting from 0 (initials)
            counts_for_hist = roi_counts_matrix[:, ::2]
        else:
            counts_for_hist = roi_counts_matrix
            
        # Plot Per-ROI Histograms
        self._plot_per_roi_histograms(counts_for_hist)

        # Global Histogram & Fitting
        flat_counts = counts_for_hist.ravel()
        popt, pcov, detection_threshold = self._fit_global_distribution(flat_counts, output_dir)

        # Calculate Fidelity and Statistics
        filling_fraction = np.sum(flat_counts > detection_threshold)/len(flat_counts)
        print(f"Filling fraction (from counts): {filling_fraction:.3f}")
        
        # Recalculate object for fidelity calc
        dg_obj = BinaryThresholding(popt) 
        fidelity = dg_obj.calculate_imaging_fidelity(filling_fraction)

        fidelity_mean_mc, fidelity_err = dg_obj.calculate_fidelity_error(pcov, filling_fraction, n_samples=5000)
        print(f"Imaging fidelity: {fidelity:.5f} ± {fidelity_err:.5f}")

        # Save Metadata
        np.savetxt(output_dir / "popt.csv", popt, delimiter=',')
        np.savetxt(output_dir / "pcov.csv", pcov, delimiter=',')
        np.savetxt(output_dir / "roi_geometry.csv", [self.nr_rows, self.nr_cols], fmt="%d", delimiter=",")
        np.save(output_dir / "filling_fraction.npy", filling_fraction)
        
        self._copy_log_file(full_import_path, output_dir)

        # Plot Global Results
        self._plot_global_fit(flat_counts, popt, detection_threshold)

        # Optional Photon Histogram
        if show_photon_hist:
            self._plot_photon_histogram(full_import_path, file_suffix, flat_counts, popt, detection_threshold)

        plt.show()

    def _plot_per_roi_histograms(self, counts_matrix):
        """Grid plot of histograms for individual ROIs"""

        nr_rois = counts_matrix.shape[0]
        fig, ax = plt.subplots(nrows=self.nr_rows, ncols=self.nr_cols, figsize=(2 * self.nr_cols, 2 * self.nr_rows),
            sharex=True, sharey=True)
        axs = np.atleast_1d(ax).ravel()

        for roi_idx in range(min(nr_rois, len(axs))):
            axs[roi_idx].hist(counts_matrix[roi_idx, :], bins=self.nr_bins_roi, edgecolor='black')
        fig.supxlabel('EMCCD Counts')
        fig.supylabel('Occurrences')
        fig.suptitle('Per-ROI Histograms')

    def _fit_global_distribution(self, counts, output_dir):
        """Fits Double Gaussian and finds threshold"""

        hist_vals, bin_edges = np.histogram(counts, bins=self.nr_bins_avg)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

        # Initial Guesses
        mean_c, std_c = np.mean(counts), np.std(counts)
        max_h = max(hist_vals)
        
        # [Amp0, Mu0, Sigma0, Amp1, Mu1, Sigma1]
        init_guess = [max_h, mean_c*0.9, std_c*0.5, max_h/4, mean_c*1.1, std_c]
        fit_limits = (0, [np.inf]*6)
        
        try:
            popt, pcov = curve_fit(FittingFunctions.double_gaussian, bin_centers, hist_vals, p0=init_guess, bounds=fit_limits)
        except RuntimeError:
            print("Warning: Curve fit failed. Using initial guess.")
            popt = init_guess
            pcov = np.zeros((6,6))

        # Calculate Threshold
        dg_obj = BinaryThresholding(popt)
        
        # Calculate derived filling fraction from fit parameters
        area_bg = popt[0]*popt[2]
        area_sig = popt[3]*popt[5]
        ff_fit = area_sig/(area_bg + area_sig)
        print(f"Filling fraction derived from fit: {ff_fit:.3f}")

        threshold = dg_obj.calculate_histogram_detection_threshold(filling_fraction=ff_fit)
        print('Detection threshold:', threshold)
        np.save(output_dir / 'detection_threshold.npy', threshold)
        return popt, pcov, threshold

    def _plot_global_fit(self, counts, popt, threshold):
        """Plots the global histogram with the double gaussian fit"""

        fig, ax = plt.subplots()
        ax.set_xlabel('EMCCD Counts')
        ax.set_ylabel('Occurrences')
        ax.hist(counts, bins=self.nr_bins_avg, edgecolor='black', label='Data')
        
        # Generate smooth fit line
        hist_vals, bin_edges = np.histogram(counts, bins=self.nr_bins_avg)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2 
        x_fit = np.linspace(bin_centers[0], bin_centers[-1], 200)
        y_fit = FittingFunctions.double_gaussian(x_fit, *popt)
        
        ax.plot(x_fit, y_fit, 'r-', label='Double Gaussian fit')
        ax.axvline(threshold, color='grey', linestyle='--', label='Detection threshold')
        ax.legend()
        ax.grid(True)
        ax.set_title("Global Histogram")

    def _plot_photon_histogram(self, import_path, suffix, counts, popt, threshold_counts):
        """Calculates and plots photon statistics (requires non-weighted recalculation)"""
        print("Calculating photon statistics...")
        backgr_counts = popt[1]
        
        # Rescale factor calculation (Weighted vs Non-Weighted)
        # We must re-calculate ROI counts without weights to get true photon numbers
        rois_obj = ROIs(self.roi_radius, self.log_thresh)
        _, counts_matrix_nw = rois_obj.calculate_roi_counts(str(import_path) + os.sep, suffix, use_weighted_count=False, roi_index_tolerance=self.roi_idx_tolerance)
        counts_nw = counts_matrix_nw.ravel()
        
        ixon = EMCCD()
        photons_weighted = ixon.counts_to_photons(counts, backgr_counts)
        photons_nw = ixon.counts_to_photons(counts_nw, backgr_counts)
        rescale_factor = photons_nw.mean() / photons_weighted.mean()
        
        # Conversion
        threshold_photons = ixon.counts_to_photons(threshold_counts, backgr_counts)
        final_photons = photons_weighted*rescale_factor
        final_threshold = threshold_photons*rescale_factor
        print(f"Photon Threshold: {final_threshold:.2f}")

        fig_width = 3.375 * 0.5 - 0.02
        fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.61))
        ax.set_xlabel('Number of photons')
        ax.set_ylabel('Probability')
        ax.grid(True)
        ax.hist(final_photons, bins=self.nr_bins_avg, edgecolor='black', density=True, label='Counts')
        ax.axvline(final_threshold, color='grey', linestyle='--', label='Threshold')
        plt.tight_layout()

    def _copy_log_file(self, src_dir, dest_dir):
        source = src_dir / 'log.csv'
        if source.exists():
            shutil.copy2(source, dest_dir / 'log.csv')
            print(f"Log file copied to {dest_dir}")
        else:
            print(f"Warning: log.csv not found in {src_dir}")
        

class SurvivalAnalysis:
    @staticmethod
    def get_survival_data(rid: str, raw_data_root: str, processed_data_root: str, roi_config: dict, 
        hist_config: dict, geometry: tuple, force_reprocess: bool = False):
        """
        Pipeline function that handles caching, processing, and loading of survival statistics.
        
        Returns:
            tuple: (x_grid, global_surv, global_sem, roi_surv, roi_sem, df)
        """
        
        # Setup Paths
        raw_path = Path(raw_data_root)
        proc_base = Path(processed_data_root)
        proc_path = proc_base / 'roi_counts' / rid
        
        # Check Cache / Run Processing
        cache_exists = (proc_path / 'roi_counts_matrix.npy').exists()
        
        if cache_exists and not force_reprocess:
            print(f"[{rid}] Found cached data. Skipping heavy processing.")
        else:
            print(f"[{rid}] Cache missing or forced. Running ROI analysis...")
            analyzer = ROICounts(
                roi_geometry=geometry,
                roi_params=roi_config,
                hist_params=hist_config,
                output_root=str(proc_base)
            )
            analyzer.process_dataset(
                images_path=str(raw_path), 
                rid=rid, 
                file_suffix='image'
            )
            print("Analysis complete.")

        # Load Data & Compute Stats
        try:
            binary_threshold = np.load(proc_path / 'detection_threshold.npy')
            df = pd.read_csv(proc_path / 'log.csv')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing processed files in {proc_path}.") from e

        sa_stats = SingleAtoms(binary_threshold, str(proc_path))
        
        # Get x-axis
        x_grid, _ = sa_stats.reshape_survival_matrix(df)
        
        # Get all statistics
        # returns: (surv_prob_per_roi, surv_prob_global, sem_per_roi, global_sem)
        stats = sa_stats.calculate_survival_statistics(df)
        
        roi_surv = stats[0]   # Shape: (n_rois, n_x_values)
        global_surv = stats[1]
        roi_sem = stats[2]    # Shape: (n_rois, n_x_values)
        global_sem = stats[3]
        
        return x_grid, global_surv, global_sem, roi_surv, roi_sem, df

    @staticmethod
    def map_rois_to_grid(geometry, missing_coords=None):
        """
        Creates a 2D grid mapping spatial positions (row, col) to the linear ROI index.
        Useful for plotting individual ROIs in their physical arrangement when some spots are missing.
        
        Args:
            geometry (tuple): (nr_rows, nr_cols)
            missing_coords (list of tuples): List of (row, col) pairs to skip. 
                e.g. [(4, 0), (2, 2)]
        
        Returns:
            np.ndarray: 2D array of shape (rows, cols). 
                Contains the ROI index at each position, or -1 if empty/missing.
        """
        nr_rows, nr_cols = geometry
        
        # Initialize grid with -1 (meaning "no ROI detected here")
        roi_grid = np.full((nr_rows, nr_cols), -1, dtype=int)
        
        # Normalize missing_coords to a list of tuples if it's a single tuple or None
        if missing_coords is None:
            missing_coords = []
        elif isinstance(missing_coords, tuple) and len(missing_coords) == 2:
            # User provided a single tuple like (4, 0) instead of [(4, 0)]
            missing_coords = [missing_coords]
            
        current_roi_idx = 0
        
        # Fill grid sequentially
        for r in range(nr_rows):
            for c in range(nr_cols):
                # Check if this specific (r, c) is in the missing list
                if (r, c) in missing_coords:
                    continue 
                
                roi_grid[r, c] = current_roi_idx
                current_roi_idx += 1
                
        return roi_grid
