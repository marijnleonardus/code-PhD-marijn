# author: Marijn Venderbosch
# november 2022

"""
This is a  script for estimating of a beam radius using knife edge method
"""

# %% imports

import pandas as pd
import math 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special
from sklearn.metrics import r2_score

# %% load and fit data

# Take data from csv file (excel)
raw_data = pd.read_csv('data.csv', delimiter='\t')

# take value from CSV file
position = raw_data['x'].values
optical_power = raw_data['power'].values


# define function for curve fitting
def error_function(x, x_center, power_background, power_max, waist):
    return power_background + (power_max / 2) * (1 - special.erf(math.sqrt(2) * (x - x_center) / waist))


# initial guesses for parameters
p0 = [10, 0, 140, 4]

# fit curve with function 
fit_params, cov = curve_fit(error_function, position, optical_power, p0)

# define the fitting function
fit_data = error_function(position,
                          fit_params[0],
                          fit_params[1],
                          fit_params[2],
                          fit_params[3])

# print beam waist
print('Beam waist =  %.2f mm'% (fit_params[3]))

# find R^2 value
print('R^2 : %.5f'%(r2_score(optical_power, fit_data)))

# %% plotting

fig, ax = plt.subplots()
ax.grid()

ax.scatter(position, optical_power)
ax.plot(position, fit_data, color='red', alpha=0.5)

fig.suptitle('Knife edge measurement')
ax.set_xlabel('Position (mm)')
ax.set_ylabel('Optical power (mW)')

plt.show()
