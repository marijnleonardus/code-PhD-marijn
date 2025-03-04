import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.constants import Boltzmann, proton_mass
from scipy.optimize import curve_fit

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from fitting_functions_class import FittingFunctions
from plotting_class import Plotting

# clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# variables
folder_name = r'T:\\KAT1\\Marijn\\thesis_measurements\\mot\\sf_time_of_flight\\second try\\'
ms=1e-3 # s
um=1e-6 # m
uK=1e-6 # K

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
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(t/ms, sx/um, yerr=error_sx/um, fmt='o', label='x', markersize=3, capsize=3, color='blue')
ax.errorbar(t/ms, sy/um, yerr=error_sy/um, fmt='o', label='y', markersize=3, capsize=3, color='orange')

x_fit = np.linspace(0, np.max(t), 1000)
ax.plot(x_fit/ms, FittingFunctions.fit_tof_data(x_fit, *popt_x)/um, label=r'fit $x$', color='darkblue')
ax.plot(x_fit/ms, FittingFunctions.fit_tof_data(x_fit, *popt_y)/um, label=r'fit $y$', color='darkorange')

ax.set_xlabel('time of flight [ms]')
ax.set_ylabel(r'$1/e$ width [$\mu$m]')
ax.legend()

Plotting.savefig(folder_name, "temperature_tof.pdf")
plt.show()

print("temperature x: ", popt_x[1]/uK, " uK") 
print("temperature y: ", popt_y[1]/uK, " uK")

