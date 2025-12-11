from scipy.constants import hbar, Boltzmann
import numpy as np

# parameters
omega = 2*np.pi*45.11*1e3      # trap frequency in rad/s
avg_n = None #0.24               # average occupation number
temperature = 2.5*10**(-6) # K


def average_n_to_temperature(average_n, omega):
    """Convert average occupation number to temperature in Kelvin

    Parameters
    ----------
    average_n : float
        Average occupation number
    omega : float
        Trap frequency in rad/s

    Returns
    -------
    temperature : float
        Temperature in Kelvin
    """
    temperature = (hbar*omega/Boltzmann)/np.log(1 + 1/average_n)
    print(f'Temperature: {temperature*1e6:.2f} uK') 


def temperature_to_average_n(temperature, omega):
    """Convert temperature to average occupation number

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    omega : float
        Trap frequency in rad/s

    Returns
    -------
    average_n : float
        Average occupation number
    """
    average_n = (np.exp(hbar*omega/Boltzmann/temperature) - 1)**(-1)
    print(f'Average occupation number: {average_n:.2f}')


def temperature_to_groundpopulation(temperature, omega):
    """Convert temperature to ground population

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin
    omega : float
        Trap frequency in rad/s

    Returns
    -------
    groundpopulation : float
        Ground population
    """
    groundpopulation = 1 - np.exp(-hbar*omega/Boltzmann/temperature)
    print(f'Ground population: {groundpopulation:.2f}')

# average_n_to_temperature(avg_n, omega)
temperature_to_average_n(temperature, omega)
temperature_to_groundpopulation(temperature, omega)