import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Path to the folder containing .tiff images
folder_path = 'Z:/Strontium/Images/2024-04-22/scan133898/'

def load_tiff_images(folder_path, filename):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return []

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Filter out only the images with filenames ending with '0000fluorescence.tif'
    tiff_files = [file for file in file_list if file.endswith(filename)]

    # Load each image and store them in a list
    images = []
    for file_name in tiff_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            image = Image.open(file_path)
            images.append(image)
            print(f"Loaded image: {file_name}")
        except Exception as e:
            print(f"Error loading image '{file_name}': {e}")
    return images

def compute_average_image(images):
    if not images:
        print("No images to compute average.")
        return None

    # Convert images to numpy arrays for easier computation
    image_arrays = [np.array(image) for image in images]

    # Compute the average image
    average_image = np.mean(image_arrays, axis=0).astype(np.uint8)
    return average_image

# Load images
loaded_images_tweezers = load_tiff_images(folder_path, '0000fluorescence.tif')
average_image_tweezers = compute_average_image(loaded_images_tweezers)

im = Image.fromarray(average_image_tweezers)
print(im)
im.save("average.tif")


plt.imshow(average_image_tweezers)
plt.show()
