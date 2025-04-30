# author: Marijn Venderbosch
# December 2022

import numpy as np
from scipy.constants import Boltzmann, proton_mass, pi


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
    def damped_sin_wave(t, A, damping_time, f, phase, offset):
        """damped sin for trap freq. measurement from thesis Labuhn p. 55
        Args:
            t (np.ndarray): time array
            A (float): 
            gamma (float): 
            f (float): 
            phase (float): 
            offset (float): 

        Returns:
            damp_sin: 
        """
        omega_trap = 2*pi*f
        damp_sin = offset + A*np.exp(-t/damping_time)*np.sin(2*omega_trap*t + phase) 
        return damp_sin
    