from scipy.constants import hbar, Boltzmann
import numpy as np

# parameters
omega = 2*np.pi*82e3      # trap frequency in rad/s
avg_n = 0.24              # average occupation number


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
    return temperature


temperature = average_n_to_temperature(avg_n, omega)
print(f'Temperature: {temperature*1e6:.2f} uK') 