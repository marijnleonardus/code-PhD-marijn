# author: Marijn Venderbosch
# July 2024

import numpy as np


class ManipulateImage:

    def crop_array(self, array, x):
        """
        Crops an array by removing `x` number of rows and columns from each side.

        Parameters:
            array (numpy.ndarray): The input array to be cropped.
            x (int): The number of rows and columns to be cropped from each side.

        Returns:
            numpy.ndarray: The cropped array. If `x` is non-positive, the input array is returned unchanged.
        """
        if x <= 0:
            # No cropping needed if x is non-positive
            return array  
        else:
            # Determine the new dimensions after cropping
            cropped_array = array[x:-x, x:-x]
            return cropped_array
