# author: Marijn Venderbosch
# December 2022

import numpy as np


class FittingFunctions:
    """collection of functions to be used for fitting (only 1 at the moment)"""
    
    def gaussian_function(x, offset, amplitude, middle, width):
        
        """returns gaussian function with standard parameters
        - offset
        - amplitude
        - middle 
        - width (sigma)"""
        
        return offset + amplitude * np.exp(-0.5 * ((x - middle) / width)**2)
