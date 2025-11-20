import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import sys
import os
import numpy as np

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../../modules', '../../utils'])

from units import nm, um
from camera_image_class import CameraImage

# constants
wavelength = 461*nm  # in meters
px_size = 3.45*um  # in meters
binning = 1  # binning factor
magnification = 150/250  # magnification factor

# data: path and image croppin settings
#path = r'Z:/Strontium/Images/2025-11-19/635229'
path = r"Z://Strontium//Images//2024-10-21//442187//"
x0 = 140
y0=475
w=150
h=150


def compute_cross_section(wavelength: float):
    """computing cross section assuming s<<1, detuning =0"""
    cross_section = 3*wavelength**2/(2*pi)
    return cross_section


def compute_od(abs_img: np.ndarray, ref_img: np, bg_img: np.ndarray, epsilon=1e-6):
    """
    Compute optical density:
        OD = -ln( (abs_img - bg_img) / (ref_img - bg_img) )
    with safe handling:
      - prevent negative or zero denominators
      - force OD >= 0 (clip ratio to <= 1)
    """
    # Background subtraction
    img_abs = abs_img.astype(float) - bg_img.astype(float)
    img_ref = ref_img.astype(float) - bg_img.astype(float)

    # Avoid negatives or zeros after subtraction
    img_abs = np.clip(img_abs, epsilon, None)
    img_ref = np.clip(img_ref, epsilon, None)

    # Compute OD. enforce OD>0 
    ratio = img_abs/img_ref
    ratio = np.clip(ratio, None, 1.0)
    optical_density = -np.log(ratio)
    return optical_density


def calculate_nr_atoms(od_image: np.ndarray):
    cross_section = compute_cross_section(wavelength)
    px_area = (px_size*binning/magnification)**2
    nr_atoms = px_area*1/cross_section*np.sum(od_image)
    return nr_atoms


def plot_with_scaled_axes(image: np.ndarray):
    pixels_y = image.shape[0]
    pixels_x = image.shape[1]
    roi_size_y = CameraImage.pixels_to_m(pixels_y, magnification, px_size, binning)
    roi_size_x = CameraImage.pixels_to_m(pixels_x, magnification, px_size, binning)
    
    fig, ax = plt.subplots(figsize=(2.5, 2.3))
    ax.set_xlabel(r'x ($\mu$m)')
    ax.set_ylabel(r'y ($\mu$m)')
    im = ax.imshow(image, cmap="inferno", extent=[0, roi_size_y/um, 0, roi_size_x/um])
    fig.colorbar(im, ax=ax, label='Optical Density')
    plt.show()


if __name__ == "__main__":
    ImageObject = CameraImage()
    
    img0 = ImageObject._load_image(path + "\\0000image.tif")   # absorption image
    img0 = ImageObject.crop_image_around_point(img0, x0=x0, y0=y0, w=w, h=h)

    img1 = ImageObject._load_image(path + "\\0001image.tif")   # reference image
    img1 = ImageObject.crop_image_around_point(img1, x0=x0, y0=y0, w=w, h=h)

    img2 = ImageObject._load_image(path + "\\0002image.tif")   # background image
    img2 = ImageObject.crop_image_around_point(img2, x0=x0, y0=y0, w=w, h=h)

    od_img = compute_od(img0, img1, img2)

    nr_atoms = calculate_nr_atoms(od_img)
    print(f"Number of atoms: {nr_atoms:.2e}")

    plot_with_scaled_axes(od_img)
