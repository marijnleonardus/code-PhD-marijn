# author: Marijn Venderbosch
# July 2024

import numpy as np
import os
import glob
from PIL import Image


class LoadImageData:
    def import_image_sequence(self, image_path, file_name_suffix):
        """Imports a sequence of images from a given path and file name suffix.

        Args:
            image_path (str): The path to the directory containing the images.
            file_name_suffix (str): The suffix that should be present in each image file name.

        Returns:
            numpy.ndarray: A 3D array representing the stack of images.
        """
        image_filenames = glob.glob(os.path.join(image_path, f"*{file_name_suffix}.tif"))

        image_stack = []
        for filename in image_filenames:
            with Image.open(filename) as img:
                image_array = np.array(img)
                image_stack.append(image_array)

        return np.array(image_stack)
    
    
class ManipulateImage:

    def crop_array_edge(self, array, crop_range_x, crop_range_y):
        """
        Crops an array by removing `x` number of rows and columns from each side.

        Parameters:
            array (numpy.ndarray): The input array to be cropped.
            crop_range_x (int): Nr of columns to be cropped from each side
            crop_range_y (int): Nr of rows to be cropped from each side

        Returns:
            cropped_array (numpy.ndarray): The cropped array. 
        """
  
        cropped_array = array[crop_range_y:-crop_range_y, crop_range_x:-crop_range_x]
        return cropped_array
    
    def crop_array_center(self, array, center_x, center_y, crop_radius):
        """crop ROI from center position (column, row)

        Args:
            array (numpy.ndarray): input array
            center_x (int): center of image (column)
            center_y (int): centerof image (row)
            crop_radius (int): amount of pixels to keep in each direction. 

        Returns:
            center_roi (numpy.ndarray): _description_
        """

        rows_start = int(center_y - crop_radius)
        rows_end = int(center_y + crop_radius)
        cols_start = int(center_x - crop_radius)
        cols_end = int(center_x + crop_radius)
        center_roi = array[cols_start:cols_end, rows_start:rows_end]
        return center_roi
