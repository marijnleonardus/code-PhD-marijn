# author: Marijn Venderbosch
# December 2022

import numpy as np


class FittingFunctions:
    """collection of functions to be used for fitting"""
    
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
        gaussian2d_angled = ampl*np.exp(-(a*(x - xo)**2 + 2*b*(x - xo)*(y - yo) + c*(y - yo)**2)) + offset
        return gaussian2d_angled
    
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
        gaussian2d = ampl*np.exp(-((x - xo)**2/(2*sigma_x**2) + (y - yo)**2)/(2*sigma_y**2)) + offset
        return gaussian2d
    
    def linear_func(x, offset, slope):
        """linear function for with offset and slope"""

        return offset + slope*x
    