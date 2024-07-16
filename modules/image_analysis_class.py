# author: Marijn Venderbosch
# July 2024

import numpy as np
from numpy import unravel_index
    

class ManipulateImage:
    """to do; need to merge some of these cropping functions that are similar
    crop_array_edge and crop_array_center can be static methods i think"""
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
        rows_end = int(center_y + (crop_radius + 1))
        cols_start = int(center_x - (crop_radius + 1))
        cols_end = int(center_x + crop_radius)
        center_roi = array[cols_start:cols_end, rows_start:rows_end]
        return center_roi
    
    @staticmethod
    def crop_to_region_of_interest(image_file, roi_size):
        """crops image file to region of interest, used for MOT images"""
    
        # Finding center MOT
        location_maximum = image_file.argmax()
        indices = unravel_index(location_maximum, image_file.shape)
    
        # Crop image                                                     
        region_of_interest = image_file[indices[0] - roi_size:indices[0] + roi_size,
            indices[1] - roi_size:indices[1] + roi_size]
    
        # Normalize but keep max. number of counts to be used in atom number formula
        max_counts = np.max(region_of_interest)
        roi_normalized = region_of_interest/max_counts
        return roi_normalized, max_counts
    
    @staticmethod
    def crop_center(img, cropx, cropy):
        """crop image centered around middle"""
        
        y, x, *_ = img.shape
        startx = x//2 - (cropx//2)
        starty = y//2 - (cropy//2)    
        return img[starty:starty + cropy, startx:startx + cropx, ...]


class Histograms():
    def weighted_count_roi(self, center_weight, pixel_box):
        """compute weighted sum of counts in a pixel box

        Args:
            center_weight (float): weight of center compared to edge pixels
            pixel_box (np array): the ROI

        Returns:
            counts (int): weighted number of counts
        """
        pixel_dim = len(pixel_box)
        center_pixel_value = pixel_box[int(0.5*(pixel_dim - 1)), int(0.5*(pixel_dim - 1))]
        nr_edge_pixels = pixel_dim**2 - 1
        
        # summing over all pixels
        # so when adding the center pixel contribution, remove 1 to avoid counting twice 
        weighted_sum = 1/(center_weight + nr_edge_pixels)*(np.sum(pixel_box) + (center_weight - 1)*center_pixel_value)
        return weighted_sum
