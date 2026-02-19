import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from modules.fitting_functions_class import FittingFunctions
from utils.units import MHz
from utils.plot_utils import Plotting

# physics params
start_time = 20000  # seconds to skip at the beginning of the data
zoom_window = 3600 # size of the zoom window in seconds (e.g., 3600 for 1 hour)

# load raw data
folder = r'T:/KAT1/Marijn/thesis_measurements/FC1500measurements/cavity_drift/2023'
file_name = 'cavity_drift_gps_originexport.csv'
path = f"{folder}/{file_name}"

# output
output_folder = r'C:/Users/s163673/OneDrive - TU Eindhoven/Documents/MyFiles/Git/personal/PhD-thesis/img/plots'

try:
    # Load the CSV file
    # Based on the image, the first 3 rows contain labels and units.
    # 'skiprows=3' starts reading the data from the first numerical row.
    # 'header=None' tells pandas that we are defining the column names ourselves.
    df = pd.read_csv(path, skiprows=3, header=None)

    # Assign columns (index 1 is the 2nd column, index 2 is the 3rd column)
    time_raw = df.iloc[:, 1]
    y_values = df.iloc[:, 2]

    # Filter: Skip the first couple thousand seconds. There is something wrong with the data here
    mask = time_raw >= start_time
    df_filtered = df[mask].copy()

    # Shift. This aligns the start of the data to t = 0
    df_filtered.iloc[:, 1] = df_filtered.iloc[:, 1] - start_time

    # Extract final X and Y for plotting
    x = df_filtered.iloc[:, 1]
    y = df_filtered.iloc[:, 2]

    # fit data and obtain slope of linear fit
    popt, pcov = curve_fit(FittingFunctions.linear_func, x, y)
    fit_line = FittingFunctions.linear_func(x, *popt)
    slope = popt[1]
    slope_error = pcov[1, 1]**0.5
    print(f"Calculated Slope: {slope} Hz/s")
    print(f"Slope Uncertainty: {slope_error} Hz/s")

    # main plot
    width = 4.9 # in
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.plot(x, y/MHz, color='blue', linewidth=1, alpha=0.5)
    ax.plot(x, fit_line/MHz, color='blue', linewidth=1)
    ax.plot(x, fit_line/MHz)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Beat frequency (Hz)')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # inset plot
    ax_ins = ax.inset_axes([0.65, 0.65, 0.3, 0.3]) #[left, bottom, width, height] in fraction of the main axes
    ax_ins.plot(x, y/MHz, color='blue', alpha=0.5)
    #ax_ins.plot(x, fit_line, color='red', linewidth=2)
    ax_ins.set_xlim(0, zoom_window)
    
    # Dynamically adjust Y-axis for the inset based on the data in that hour
    hour_mask = (x >= 0) & (x <= zoom_window)
    if any(hour_mask):
        y_hour = y[hour_mask]
        ax_ins.set_ylim(np.min(y_hour/MHz), np.max(y_hour/MHz))
    ax_ins.grid(True, linestyle=':', alpha=0.5)

    Plot=Plotting(output_folder)  
    Plot.savefig('cavity_drift_with_gps.pdf')

    plt.show()

except FileNotFoundError:
    print(f"Error: The file at {path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
