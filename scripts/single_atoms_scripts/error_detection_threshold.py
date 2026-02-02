## author: Marijn Venderbosch
## January 2026

"""
Script to calculate the error in the imaging fidelity,
using the script `histogram_and_treshold.py`
"""

import numpy as np
from scipy.stats import norm, multivariate_normal

# append modules dir
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from single_atoms_class import BinaryThresholding

# obtain double gaussian fit parameters, detection treshold, filling fraction
# saved by `histogram_and_treshold.py`
images_path = 'Z:\\Strontium\\Images\\2026-01-23\\scan172311\\'
filling_fraction = np.load(images_path + 'filling_fraction.npy')
x_t = np.load(images_path + 'filling_fraction.npy')
params_best = np.loadtxt(images_path + 'popt.csv', delimiter=',')

# obtain errors in double gaussian fit
fit_cov_matrix = np.loadtxt(images_path + 'pcov.csv', delimiter=',')
covariance = np.diag(fit_cov_matrix)
perr = np.sqrt(covariance)

# Monte Carlo Loop
N_SAMPLES = 10000
fidelities = []
valid_samples_count = 0

# Generate random parameter sets (size 6)
samples = multivariate_normal.rvs(mean=params_best, cov=covariance, size=N_SAMPLES)

print(f"Running Monte Carlo on {N_SAMPLES} samples...")

for params in samples:
    # Unpack for readability
    a0_s, m0_s, s0_s, a1_s, m1_s, s1_s = params
    
    # Physical constraints check
    if s0_s <= 0 or s1_s <= 0 or a0_s <= 0 or a1_s <= 0:
        continue

    # 1. Get Threshold using YOUR function
    # Pass the full 6-element array and p1
    ArbitraryDoubleGauss = BinaryThresholding(params)
    xt_result = ArbitraryDoubleGauss.calculate_histogram_detection_threshold(filling_fraction=filling_fraction)
    
    # Handle cases where no solution was found in range
    if len(xt_result) == 0:
        continue
        
    # Extract the scalar value (since valid_sol is a list/array)
    xt_scalar = xt_result[0]

    # 2. Calculate Fidelity
    # Note: calculate_fidelity needs means/sigmas, not amplitudes
    f_s = ArbitraryDoubleGauss.calculate_imaging_fidelity(filling_fraction=filling_fraction)
    fidelities.append(f_s)
    valid_samples_count += 1

fidelities = np.array(fidelities)
fid_mean = np.mean(fidelities)
fid_std_err = np.std(fidelities)

# Best fit check
BestDoubleGauss = BinaryThresholding(params_best)
xt_best_arr = BestDoubleGauss.calculate_histogram_detection_threshold(filling_fraction=filling_fraction)
xt_best = xt_best_arr[0] if len(xt_best_arr) > 0 else 0
fid_best = BestDoubleGauss.calculate_imaging_fidelity(filling_fraction=filling_fraction)

print("-" * 30)
print(f"Valid Samples processed:  {valid_samples_count}")
print(f"Optimal Threshold (Best): {xt_best}")
print(f"Best Fit Fidelity:        {fid_best:.5f}")
print(f"Mean MC Fidelity:         {fid_mean:.5f}")
print(f"Standard Error:           {fid_std_err:.5f}")
print("-" * 30)