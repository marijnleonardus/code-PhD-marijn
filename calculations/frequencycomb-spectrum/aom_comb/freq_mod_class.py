# author: Marijn Venderbosch
# July 2023

from scipy.constants import pi
import numpy as np
from scipy import signal


class FrequencyModulation:

    def __init__(self, mod_freq: float, mod_depth: float, carrier_freq: float): 
        self.mod_freq = mod_freq
        self.mod_depth = mod_depth
        self.carrier_freq = carrier_freq

    def sinusoidal_modulation_signal(self, t: np.ndarray) -> np.ndarray:
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
                
        # generate sawtooth signal
        sawtooth = -0.5*signal.sawtooth(2*pi*self.mod_freq*t) - 0.5

        # generate freq. as function of time with correct height, width
        ramp_signal = self.carrier_freq + self.mod_depth*sawtooth
        return ramp_signal
    
    def linear_modulation_signal(self, t: np.ndarray) -> np.ndarray:
        
        



