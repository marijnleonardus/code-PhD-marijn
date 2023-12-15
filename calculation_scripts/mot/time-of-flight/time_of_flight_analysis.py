import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import proton_mass, Boltzmann

"""script to analyze time of flight data from camera images
tries to extract magnification factor (unsuccesful) and temperature"""

# %% parameters

sr_mass = 88*proton_mass
image_distance = 6.389
pixel_size = 3.45e-6
bin_size = 2

magnification = 0.218
cam_pixelsize = bin_size * pixel_size

# %% load data

file_location = "T:\\KAT1\\Marijn\\redmot\\time of flight\\"
file_name = "bb_tof"
df = pd.read_csv(file_location + file_name)
arr = df.to_numpy()

locations_pixels = arr[:, 3]
times = arr[:, 0]*0.001
radii = .5*(arr[:,1]+arr[:,2])*pixel_size/magnification


# %% fit cloud location gravity

def gravity_acceleration(time, starting_position, a):
    position = starting_position -a*0.5* 9.8 * time**2
    return position


heights = -locations_pixels * cam_pixelsize/magnification

dropping_guess = [arr[0, 3], 1e4]
dropping_params, _ = curve_fit(gravity_acceleration, times, heights, p0=dropping_guess)
dropping_fitcurve = gravity_acceleration(times, *dropping_params)

# %% temperature fit

def thermal_expansion(time, r0, expansion_velocity):
    radius = np.sqrt(r0**2 + expansion_velocity**2*time**2)
    return radius


expansion_guess = [5e-3, 20e-3]
expansion_params, _ = curve_fit(thermal_expansion, times, radii, p0=expansion_guess) 
expansion_fitcurve = thermal_expansion(times, *expansion_params)

sigma_v = expansion_params[1]
temperature = sr_mass * sigma_v**2 / Boltzmann
print(temperature)

# %% plotting

fig, ax = plt.subplots()
ax.scatter(times, heights)
ax.plot(times, dropping_fitcurve, 'r-')

fig, ax = plt.subplots()
ax.scatter(times, radii)
ax.plot(times, expansion_fitcurve, 'r-')
ax.set_xlabel('time of flight [s]')
ax.set_ylabel('1/e radius [m]')
plt.show()
