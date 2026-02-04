# author marijn Venderbosch
# january 2026

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# User defined libraries
from modules.roi_analysis_class import ROICounts
from modules.fitting_functions_class import FittingFunctions
from utils.units import us

os.system('cls' if os.name == 'nt' else 'clear')

# raw data and processed data locations
rid = 'scan184404' # rydberg ramsey
raw_path = 'Z:\\Strontium\\Images\\2026-01-29\\'
processed_root = 'output/processed_data/'

# ROI Analysis Settings (Only needed if running analysis from scratch)
roi_config = {
    'radius': 2,
    'log_thresh': 10,
    'index_tolerance': 5
}
hist_config = {
    'nr_bins_roi': 30,
    'nr_bins_avg': 60,
    'plot_only_initial': True
}

# ROI geometry. If the 4th c in the 2nd row is missing do: 1,3
geometry = (5, 5) # rows, cols
missing_row, missing_col = None, None

# Initialize the analyzer
analyzer = ROICounts(
    roi_geometry=geometry,
    roi_params=roi_config,
    hist_params=hist_config,
    output_root=processed_root
)

# Run the processing
# This will:
# 1. Calculate counts from raw images
# 2. Fit the Double Gaussian
# 3. Calculate Detection Threshold & Fidelity
# 4. Show the Histogram plots
analyzer.process_dataset(
    images_path=raw_path, 
    rid=rid, 
    file_suffix='image',
    show_photon_hist=False # Set to True if you want the photon-number x-axis
)