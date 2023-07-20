# author: Marijn Venderbosch
# July 2023

from scipy.constants import pi
import numpy as np
from scipy import signal
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt


class FrequencyModulation:

    def __init__(self, mod_freq: float, mod_depth: float, carrier_freq: float): 
        self.mod_freq = mod_freq
        self.mod_depth = mod_depth
        self.carrier_freq = carrier_freq

    def sinusoidal_modulation(self, t: np.ndarray) -> np.ndarray:
        """returns frequency modulated signal for sinusoidal modulation 

        Args:
            t (np.ndarray): time matrix

        Returns:
            fm_signal (np.ndarray): frequency modulated signal
        """
        phase = 2*pi*self.carrier_freq*t + self.self.mod_depth *np.sin(2*pi*self.self.mod_depth*t)
        fm_signal = np.cos(phase)
        return fm_signal

    def linear_frequency_ramp(self, t: np.ndarray) -> np.ndarray:
        """computes frequency modulated signal for linear frequency ramp

        Args:
            t (np.ndarray): time, matrix

        Returns:
            np.ndarray: frequency as a function of time
        """
        # generated modulation signal from sawtooth function
        mod_signal = -(0.5*signal.sawtooth(2*pi*self.mod_freq*t)+0.5)
        
        # frequency modulation, formula from wiki page
        # because 'cumtrapz' function does not work on first entry, 
        # remove first entry of 't' matrix
        integral = cumtrapz(mod_signal, t)
        phase = 2*pi*self.carrier_freq*t[1::] + 2*pi*self.mod_depth*integral
        fm_signal = np.cos(phase)
        return fm_signal
    