# author: Marijn Venderbosch
# December 2022

import numpy as np
from PIL import Image
import os
import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


class CameraImage:
    """Collection of functions to do with loading and processing camera images"""

    def _load_image(self, filename):
        with Image.open(filename) as img:
            return np.array(img)
        
    def import_image_sequence(self, image_path, file_name_suffix):
        """Imports a sequence of images from a given path and file name suffix
        make faster using executor.map

        Args:
            image_path (str): The path to the directory containing the images.
            file_name_suffix (str): The suffix that should be present in each image file name.

        Returns:
            numpy.ndarray: A 3D array representing the stack of images.
        """

        image_filenames = glob.glob(os.path.join(image_path, f"*{file_name_suffix}.tif"))

        # Use multithreading for faster image loading
        with ThreadPoolExecutor(max_workers=8) as executor:
            image_stack = list(executor.map(self._load_image, image_filenames))
        return np.array(image_stack)
    
    @staticmethod
    def load_image_from_file(location, name):
    
        """spits out numpy array of BMP image loaded into memory"""
        
        # load image and convert to greyscale
        image_file = Image.open(location + name)
        #image_file_grey = image_file.convert("I;16") 
        #for 8 bit 
        image_file_grey = image_file.convert("L")

        # convert to numpy format
        array = np.array(image_file_grey)
        return array
    @staticmethod
    def compute_pixel_sums_x_y(image_file):
        """computes pixel sums over rows (y) and columns (x) of given image"""
        
        sum_rows = image_file.sum(axis=1)
        sum_cols = image_file.sum(axis=0)
        
        # normalize
        sum_rows = sum_rows/np.max(sum_rows)
        sum_cols = sum_cols/np.max(sum_cols)
        return sum_rows, sum_cols

    @staticmethod
    def pixels_to_m(nr_pixels, magnification, pixel_size, bin_size):
        """converts number of pixels to meters"""

        image_size = nr_pixels*pixel_size*bin_size
        object_size = image_size/magnification
        return object_size
    
    @staticmethod
    def m_to_pixels(object_size, magnification, pixel_size, bin_size):
        """converts meters to number of pixels"""

        image_size = object_size*magnification
        nr_pixels = image_size/bin_size/pixel_size
        return nr_pixels


class EMCCD:
    def __init__(self):
        """Initialize the EMCCD with counts and background counts.

        Args:
            emccd_counts (float): EMCCD counts.
            background_counts (float): Background counts.
        """
        self.em_gain = 300
        self.quantum_eff = 0.8
        self.sensitivity = 3.68

    def counts_to_photons(self, emccd_counts, background_counts):
        """convert EMCCD counts to photons using the relation from EMCCD manual

        Args:
            emccd_counts (float): EMCCD counts

        Returns:
            photons (float): number of photons
        """
        nr_photons = (emccd_counts - background_counts)*self.sensitivity/(self.em_gain*self.quantum_eff)
        return nr_photons
