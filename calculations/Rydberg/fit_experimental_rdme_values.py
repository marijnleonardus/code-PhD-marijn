# author: Robert de Keijzer, Marijn Venderbosch
# January 2022

"""
script for fitting experimental data for radial dipole matrix elements
this is for 3P1 - 3S1, which are equivalent to 3P0 - 3S1 up to a Clebsch Gordan coefficient
in this case 1/sqrt(3)

I copied and slightly edited this script from Robert de keijzer
"""

# %% Imports

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.constants import c

# user defined
from classes.conversion_class import Conversion
from classes.optics_class import Optics


# %% variables

pi = np.pi
madjarov_power = 30e-3  # [W]
madjarov_waist = 18e-6  # [m]
madjarov_rabi = 2*pi*12e6  # [Hz]
wavelength_rydberg = 317e-9  # [m]


def fit_gr_dependence(x, a):
    """
    fits n^(-1.5) dependence of ground-rydberg RDME for n^prime = n - defect

    Parameters
    ----------
    x : array
        indepdenent variable.
    a : float
        fit parameter.

    Returns
    -------
    TYPE
        function that describes n^-1.5 behavior of rydberg state.
    """
    
    # quantum defect for 3S1 series, large n
    defect = 3.371
    return a*(x-defect)**(-1.5)

# %% importing and fitting data

# Data from  Canzhu Tan et al 2022 Chinese Phys. Lett. 39 093202
# These are RDME values between 5s5p3P1 and 5s5n3S1
n_values = np.arange(19, 41)
rdme_values = genfromtxt("calculations/rydberg/data/data_Tan2022.csv", delimiter=',')

# fit data, guess is slope is 2.0
popt_rdme, _ = curve_fit(fit_gr_dependence, n_values, rdme_values, p0=[2.0])

# generate extrapolation by extenting fit data to higher n
n_values_plot = np.arange(19, 70)

rdme_values_fit= fit_gr_dependence(n_values_plot, *popt_rdme)

# convert to einstein coefficients
# transition frequency 
omega21 = Conversion.wavelength_to_freq(wavelength_rydberg)

# einstein coefficients data points
einstein_coefficients = Conversion.rdme_to_rate(rdme_values, 0, omega21, 0)

# einstein coefficients fit
einstein_coefficients_fit = Conversion.rdme_to_rate(rdme_values_fit, 0, omega21, 0)

# %% Plot result

plt.scatter(n_values, rdme_values)
plt.plot(n_values_plot, rdme_values_fit)
plt.xlabel('$n$')
plt.ylabel('RMDE [a.u.]')


# %% Print result for n=61 to check with madjarov data

# print n=61:
index_61 = np.where(n_values_plot==61)
rate_61 = einstein_coefficients_fit[index_61]
print('our computed linewidth: 2pi times ' + str(rate_61/(2*pi)) + ' Hz')

madjarov_intensity = Optics.gaussian_beam_intensity(madjarov_waist, madjarov_power)
omega21 = 2*pi*c/wavelength_rydberg
rate_from_madjarov = Conversion.rabi_freq_to_rate(madjarov_intensity, madjarov_rabi, omega21)
print('value from madjarov: 2pi times: ' + str(rate_from_madjarov/(2*pi)) + ' Hz')

# save result to be used later
np.savetxt('calculations/Rydberg/data/n_values.csv', n_values, delimiter=',')
np.savetxt('calculations/Rydberg/data/n_values_plot.csv', n_values_plot, delimiter=',')
np.savetxt('calculations/Rydberg/data/rdme_values.csv', rdme_values, delimiter=',')
