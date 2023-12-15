# author: Marijn Venderbosch
# December 2022

import numpy as np
from numpy import unravel_index
from PIL import Image
from decimal import Decimal


class CameraImage:
    """Collection of functions to do with loading and processing camera iamges"""
    
    def load_image_from_file(location, name):
        """spits out numpy array of BMP image loaded into memory"""
        
        # load image and convert to greyscale
        image_file = Image.open(location + name)
        image_file_grey = image_file.convert("L")

        # convert to numpy format
        array = np.array(image_file_grey)
        return array

    def crop_to_region_of_interest(image_file, roi_size):
        """crops image file to region of interest"""
    
        # Finding center MOT
        location_maximum = image_file.argmax()
        indices = unravel_index(location_maximum, image_file.shape)
    
        # Crop image                                                     
        region_of_interest = image_file[indices[0] - roi_size:indices[0] + roi_size,
                                        indices[1] - roi_size:indices[1] + roi_size]
    
        # Normalize but keep max. number of counts to be used in atom number formula
        max_counts = np.max(region_of_interest)
        roi_normalized = region_of_interest / max_counts
        return roi_normalized, max_counts
    
    def crop_center(img, cropx, cropy):
        """crop image centered around middle"""
        
        y, x, *_ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)    
        return img[starty:starty + cropy, startx:startx + cropx, ...]

    def compute_histograms_x_y(image_file):
        """computes histogram over rows (y) and columns (x) of given image"""
        
        histogram_rows = image_file.sum(axis=1)
        histogram_cols = image_file.sum(axis=0)
        
        # normalize
        histogram_rows = histogram_rows / np.max(histogram_rows)
        histogram_cols = histogram_cols / np.max(histogram_cols)
        return histogram_rows, histogram_cols

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