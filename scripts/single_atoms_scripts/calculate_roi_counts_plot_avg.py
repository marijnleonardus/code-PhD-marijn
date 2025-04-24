# author: Marijn Venderbosch
# july 2024 - April 2025

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import blob_log

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from camera_image_class import CameraImage
from single_atoms_class import ROIs

os.system('cls' if os.name == 'nt' else 'clear')

# variables
images_path = 'Z:\\Strontium\\Images\\2025-04-17\\scan131340\\'  # path to images
file_name_suffix = 'image'  # import files ending with image.tif


def calculate_roi_counts(images_path, file_name_suffix):
    # variables
    rois_radius = 2  # ROI size. Radius 1 means 3x3 array
    log_threshold = 10 # laplacian of gaussian kernel sensitivity
    weight_center_pixel = 1

    # images without cropping ('raw' data)
    image_stack = CameraImage().import_image_sequence(images_path, file_name_suffix)
    images_list = [image_stack[i] for i in range(image_stack.shape[0])]

    if np.shape(image_stack)[0] == 0:
        raise ValueError("No images loaded, check image path and file name suffix")
    else:
        print("nr images, pixels, pixels", np.shape(image_stack))

    # detect laplacian of gaussian spot locations from avg. over all images
    z_project = np.mean(image_stack, axis=0)
    spots_LoG = blob_log(z_project, max_sigma=3, min_sigma=1, num_sigma=3, threshold=log_threshold)
    y_coor = spots_LoG[:, 0] 
    x_coor = spots_LoG[:, 1]
    print(spots_LoG)
    print("nr spots detected", np.shape(spots_LoG)[0])

    # plot average image and mark detected maximum locations in red, check if LoG was correctly detected
    fig1, ax1 = plt.subplots()
    ax1.imshow(z_project)
    ax1.scatter(x_coor, y_coor, marker='x', color='r')
    fig1.show()
    ax1.set_title('Average image and LoG detected spots')

    # compute nr of counts in each ROI 
    ROIcounts = ROIs(rois_radius, weight_center_pixel)
    image_stack = np.stack(images_list, axis=0)    # shape: (n_images, H, W)
    rois_matrix, roi_counts_matrix = ROIcounts.compute_pixel_sum_counts(
        image_stack, y_coor, x_coor
    )
    # plot average pixel box for ROI 1 to check everything went correctly
    ROIcounts.plot_average_of_roi(rois_matrix[0, :, :, :])
    plt.show()

    # (nr_rois, nr_images)
    return roi_counts_matrix


def main():
    roi_counts_matrix = calculate_roi_counts(images_path, file_name_suffix)


if __name__ == "__main__":
    main()
