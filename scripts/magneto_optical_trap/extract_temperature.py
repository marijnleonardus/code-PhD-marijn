import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.constants import Boltzmann, proton_mass
from scipy.optimize import curve_fit

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

# user defined libraries
from fitting_functions_class import FittingFunctions
from plot_utils import Plotting
from units import ms, um, uK

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
folder_name = r'T:\\KAT1\\Marijn\\thesis_measurements\\mot\\sf_time_of_flight\\second try\\'

# load data from origin file
data_x = np.genfromtxt(folder_name + "logresult_sizexprocessed.csv", delimiter=',')
data_y = np.genfromtxt(folder_name + "logresult_sizeyprocessed.csv", delimiter=',')

# exclude last datapoints
nr_datapoints = 11
data_x = data_x[:nr_datapoints]
data_y = data_y[:nr_datapoints]

t = data_x[:, 0]
sx = data_x[:, 1]
sy = data_y[:, 1]
error_sx = data_x[:, 2]
error_sy = data_y[:, 2]

fit_guess = [200e-6, 5e-6]
popt_y, pcov_y = curve_fit(FittingFunctions.fit_tof_data, t, sy, sigma=error_sy, p0=fit_guess)
popt_x, pcov_x = curve_fit(FittingFunctions.fit_tof_data, t, sx, sigma=error_sx, p0=fit_guess)

# plot
fig_width = 3.375  # inches, matches one column
fig_height = fig_width*0.61
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.errorbar(t/ms, sx/um, yerr=error_sx/um, fmt='o', markersize=3, capsize=3, color='blue')
ax.errorbar(t/ms, sy/um, yerr=error_sy/um, fmt='o', markersize=3, capsize=3, color='orange')

x_fit = np.linspace(0, np.max(t), 1000)
ax.plot(x_fit/ms, FittingFunctions.fit_tof_data(x_fit, *popt_x)/um, color='darkblue')
ax.plot(x_fit/ms, FittingFunctions.fit_tof_data(x_fit, *popt_y)/um, color='darkorange')

ax.set_xlabel('Time of flight [ms]')
ax.set_ylabel(r'Gaussian fit $\sigma$ [$\mu$m]')

Plot = Plotting('output')
Plot.savefig("temperature_tof.pdf")
plt.show()

print("temperature x: ", popt_x[1]/uK, "pm", np.sqrt(pcov_x[1,1])/uK, " uK") 
print("temperature y: ", popt_y[1]/uK, " uK", np.sqrt(pcov_y[1,1])/uK, " uK")
