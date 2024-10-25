# author: Marijn Venderbosch
# January 2023

import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# %% import data

df = pd.read_csv('plots/phase noise/frequencynoise.csv', delimiter=';', decimal=',', header=None)
picoscope_data = np.array(df)

omega = picoscope_data[:, 0]
noise = picoscope_data[:, 1]

# %% functions


def window_function_1(r, w, t):
    """
    windows function that selects relevant of frequency spectrum as a function of time
    see thesis Anant Kale, p. 23

    Parameters
    ----------
    r : float
        rabi frequency in [Hz].
    w : float
        frequency, independent variable (omega) [Hz].
    t : float
        time in [s].

    Returns
    -------
    window : array
        spectral window function that selects parts of spectrum.

    """
    
    # compute numerator
    term1=3*r**2+w**2+(r**2-w**2)*np.cos(2*r*t)
    term2=-4*r**2*np.cos(r*t)*np.cos(w*t)-4*r*w*np.sin(r*t)*np.sin(w*t)
    numerator = term1+term2
    
    # compute denominator and fraction
    denominator = (r**2-w**2)**2*t**2
    window = numerator/denominator
    return window 


def product_function_1(rabi, omega):
    """
    computes product of window function with frequency noise spectrum
    so that this function can be integrated at time t1=pi/rabi

    Parameters
    ----------
    rabi : float
        rabi frequency in [Hz].
    omega : float
        frequency in [Hz].

    Returns
    -------
    product : array
        window function times noise spectrum

    """
    t1 = np.pi/rabi
    window1t = window_function_1(rabi, omega, t1)
    noise_spectrum = noise
    
    product = window1t * noise_spectrum
    return product


def compute_fidelity_error(rabi):
    """
    computes fidelity error as result of phase noise 
    see thesis Anant Kale eq. 4.24 

    Parameters
    ----------
    rabi : float
        rabi frequency in [Hz].

    Returns
    -------
    fidelity_error : float
        error in pi pulse fidelity.

    """
    # compute product function for specific rabi frequency and frequencies array
    product1t = product_function_1(rabi, omega)
    
    # integrate this spectrum
    integral = simpson(product1t, omega)
    
    fidelity_error = np.pi**2/(16*rabi**2)*1/(2*np.pi)*integral
    return fidelity_error

 
# %% compute fidelity error as a function of rabi frequency




# %% plotting

# plot frequency noise spectrum
fig, ax = plt.subplots()
ax.plot(omega, noise)

# plot noise multiplied with window function for example rabi freq of 5 MHz
product1t = product_function_1(5e6,omega)
fig2, ax2 = plt.subplots()
ax2.plot(omega, product1t)
