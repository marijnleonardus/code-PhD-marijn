# author: Marijn Venderbosch
# December 2022

import numpy as np
from numpy import unravel_index
from PIL import Image
import os
import glob
from PIL import Image


class CameraImage:
    """Collection of functions to do with loading and processing camera images"""
    
    def load_image_from_file(location, name):
        """spits out numpy array of BMP image loaded into memory"""
        
        # load image and convert to greyscale
        image_file = Image.open(location + name)
        image_file_grey = image_file.convert("L")

        # convert to numpy format
        array = np.array(image_file_grey)
        return array

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

    def compute_pixel_sums_x_y(image_file):
        """computes pixel sums over rows (y) and columns (x) of given image"""
        
        sum_rows = image_file.sum(axis=1)
        sum_cols = image_file.sum(axis=0)
        
        # normalize
        sum_rows = sum_rows/np.max(sum_rows)
        sum_cols = sum_cols/np.max(sum_cols)
        return sum_rows, sum_cols

    def pixels_to_m(nr_pixels, magnification, pixel_size, bin_size):
        """converts number of pixels to meters"""

        image_size = nr_pixels*pixel_size*bin_size
        object_size = image_size/magnification
        return object_size
    
    def m_to_pixels(object_size, magnification, pixel_size, bin_size):
        """converts meters to number of pixels"""

        image_size = object_size*magnification
        nr_pixels = image_size/bin_size/pixel_size
        return nr_pixels
