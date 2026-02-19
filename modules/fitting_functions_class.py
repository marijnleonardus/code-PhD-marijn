# author: Marijn Venderbosch
# December 2022

import numpy as np
from scipy.constants import Boltzmann, proton_mass, pi
from scipy.optimize import curve_fit


class FittingFunctions:
    """collection of functions to be used for fitting"""
    
    @staticmethod
    def gaussian_function(x, offset, amplitude, middle, width):
        """returns gaussian function with standard parameters

        arguments:
        - x (np.array): input data
        - offset (float): offset from y=0
        - amplitude(float): the amplitude of the gaussian
        - middle (float): x0
        - width (float): sigma
        
        returns: 
        - gaussian1d (np.array): gaussian function
        """
        gaussian1d = offset + amplitude*np.exp(-0.5*((x - middle)/width)**2)
        return gaussian1d

    @staticmethod
    def gaussian_2d_angled(xy, ampl, xo, yo, sigma_x, sigma_y, theta, offset):
        """2D Gaussian function that may rotate with xy plane (theta angle)

        Arguments:
        - xy (2d np array): 2D array containing x and y coordinates.
        - ampl (float): Amplitude of the Gaussian.
        - xo (float): x-coordinate of the center.
        - yo(float): y-coordinate of the center.
        - sigma_x (float): Standard deviation along the x-axis.
        - sigma_y (float): Standard deviation along the y-axis.
        - theta (float): Rotation angle of the Gaussian in radians.
        - offset (float): Offset or background.

        Returns:
        - Values of the Gaussian function evaluated at given coordinates.
        """
        x, y = xy
        a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
        b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
        c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
        
        # Gaussian function
        exponent = -(a*(x - xo)**2 + 2*b*(x - xo)*(y - yo) + c*(y - yo)**2)
        gaussian2d_angled = ampl*np.exp(exponent) + offset
        return gaussian2d_angled
    
    @staticmethod
    def gaussian_2d(xy, ampl, xo, yo, sigma_x, sigma_y, offset):
        """2D Gaussian function with no theta (angled) dependence

        Arguments:
        - xy (2d np array): 2D array containing x and y coordinates.
        - ampl (float): Amplitude of the Gaussian.
        - xo (float): x-coordinate of the center.
        - yo(float): y-coordinate of the center.
        - sigma_x (float): Standard deviation along the x-axis.
        - sigma_y (float): Standard deviation along the y-axis.
        - offset (float): Offset or background.

        Returns:
        - Values of the Gaussian function evaluated at given coordinates.
        """
        x, y = xy
        
        # Gaussian function
        exponent = (x - xo)**2/(2*sigma_x**2) + (y - yo)**2/(2*sigma_y**2)
        gaussian2d = ampl*np.exp(-1*exponent) + offset
        return gaussian2d
   
    @staticmethod
    def double_gaussian(x, amplitude1, mu1, sigma1, amplitude2, mu2, sigma2):
        """double gaussian function, for histogram fit for example
        
        arguments:
        - x (np.array): input data
        - amplitude1 (float): amplitude of first gaussian    
        - mu1 (float): mean of first gaussian                  
        - sigma1 (float): standard deviation of first gaussian
        - amplitude2 (float): amplitude of second gaussian
        - mu2 (float): mean of second gaussian
        - sigma2 (float): standard deviation of second gaussian

        returns:
        - sum_gauss (np.array): double gaussian function
        """
        gauss1 = amplitude1*np.exp(-0.5*((x - mu1) /sigma1)**2)
        gauss2 = amplitude2*np.exp(-0.5*((x - mu2) /sigma2)**2)
        sum_gauss = gauss1 + gauss2
        return sum_gauss
    
    @staticmethod
    def linear_func(x, offset, slope):
        """linear function for with offset and slope

        Args:
            x (np array): independent variable
            offset (float): y axis intercept
            slope (float): 

        Returns:
            y (np array): dependent variable
        """
        y = offset + slope*x
        return y
    
    @staticmethod
    def fit_tof_data(t, sigma_0, T):
        """function of the form sqrt(sigma_0^2 + k_b T/m * t^2)"""
        sr_mass = 88*proton_mass
        sigma = np.sqrt(sigma_0**2 + Boltzmann*T/sr_mass*t**2)
        return sigma
        
    @staticmethod
    def lorentzian(x, offset, amplitude, middle, width):
        """returns lorentzian function with standard parameters

        arguments:
        - x (np.array): input data
        - offset (float): offset from y=0
        - amplitude(float): the amplitude of the lorentzian
        - middle (float): x0
        - width (float): gamma
        
        returns: 
        - lorentzian1d (np.array): lorentzian function
        """
        lorentzian1d = offset + amplitude*width**2/((x - middle)**2 + width**2)
        return lorentzian1d
    
    @staticmethod
    def damped_sin_wave(t, ampl: float, damping_time: float, freq: float, phase: float, offset: float): 
        """damped sin in both amplitude, and offset (T1 decay involved)"""

        omega = 2*pi*freq
        damped_sin = (ampl*np.sin(omega*t + phase) + offset) *np.exp(-t/damping_time)
        return damped_sin

    @staticmethod
    def dephasing_sin_exponential(t, contrast, tau, freq, phase, offset):
        """Sine wave with exponential amplitude decay, constant offset.
        see Ivo thesis p.75"""
        
        omega = 2*np.pi*freq
        return contrast*(0.5 + 0.5*np.exp(-t/tau)*np.sin(omega*t + phase)) + offset
    
    @staticmethod
    def dephasing_sin_gaussian(t, ampl, tau, freq, phase, offset):
        """Sine wave with Gaussian amplitude decay (Inhomogeneous dephasing)."""
        
        omega = 2*np.pi*freq
        # tau here represents the 1/e width of the envelope
        return ampl*np.exp(-(t/tau)**2)*np.sin(omega*t + phase) + offset
    
    @staticmethod
    def get_model(model_name):
        """Returns the function handle associated with a model name"""
        
        mapping = {
            'full_decay': FittingFunctions.damped_sin_wave,
            'dephasing_exponential': FittingFunctions.dephasing_sin_exponential,
            'dephasing_sin_gaussian': FittingFunctions.dephasing_sin_gaussian
        }
        # Returns the requested model, or defaults to Gaussian if not found
        return mapping.get(model_name, FittingFunctions.dephasing_sin_gaussian)
    

class FitRabiOscillations:
    def __init__(self, x, y, yerr, tau, bounds=None, model='damped_sin_wave'):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.tau_guess = tau
        self.bounds = bounds if bounds is not None else (-np.inf, np.inf) 
        self.model = model

    def _estimate_fit_params(self):
        offset_guess = np.nanmean(self.y)
        ampl_guess = (np.nanmax(self.y) - np.nanmin(self.y)) / 2

        # Frequency Guess using FFT (ensure data is centered)
        y_centered = self.y - offset_guess
        n = len(self.x)
        # Use the average spacing if not perfectly uniform
        dt = np.mean(np.diff(self.x))
        
        fft_values = np.abs(np.fft.rfft(y_centered))
        frequencies = np.fft.rfftfreq(n, d=dt)
        
        # Ignore DC and low-frequency noise
        peak_idx = np.argmax(fft_values[1:]) + 1
        freq_guess = frequencies[peak_idx]

        # Phase Guess: Calculate phase at the first data point
        # y = A*sin(w*t + phi) + offset => phi = arcsin((y-offset)/A) - w*t
        try:
            w_guess = 2 * np.pi * freq_guess
            val = (self.y[0] - offset_guess) / ampl_guess
            # Clip to avoid math domain errors
            phase_guess = np.arcsin(np.clip(val, -1, 1)) - w_guess * self.x[0]
        except:
            phase_guess = 0

        return ampl_guess, freq_guess, offset_guess, phase_guess
    
    def perform_fit(self):
        try:
            ampl, freq, offset, phase_guess = self._estimate_fit_params()
            p0 = [ampl, self.tau_guess, freq, phase_guess,  offset]

            func = FittingFunctions.get_model(self.model)

            sigma = None
            if self.yerr is not None:
                sigma = np.clip(self.yerr, 1e-12, None)

            sigma = np.asarray(self.yerr, dtype=float)
            sigma[~np.isfinite(sigma)] = np.nan
            sigma = np.nan_to_num(sigma, nan=np.nanmedian(sigma))
            sigma = np.clip(sigma, 1e-6, None)

            popt, pcov = curve_fit(func, self.x, self.y, p0=p0, sigma = sigma,
                bounds=self.bounds if self.bounds is not None else (-np.inf, np.inf),
                method='trf', maxfev=500000
            )

            return popt, pcov

        except Exception as e:
            print(f"Fit failed: {e}")
            return None
        