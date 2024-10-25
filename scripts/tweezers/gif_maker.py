# author; Marijn Venderbosch
# October 2024

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# append path with 'modules' dir in parent folder
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined libraries
from camera_image_class import CameraImage

# variables
raw_data_path = "Z://Strontium//Images//2024-10-10//scan114820"  
raw_data_suffix = 'image'
output_filename = "output.gif"

image_stack = CameraImage().import_image_sequence(raw_data_path, raw_data_suffix)

# Normalize the frames to the range [0, 255] to avoid issues with the conversion to 8 bit
# this is because in the original image the values are already in close to 0 on a 32 bit scale
image_stack = (image_stack - np.min(image_stack)) / (np.max(image_stack) - np.min(image_stack)) * 2**8
image_stack = image_stack.astype('uint8')
plt.imshow(image_stack[1])
plt.show()

# Create a list to hold the images
images = []

# Loop through each 2D array in the 3D array
for i in range(image_stack.shape[0]):
    # Convert the 2D array to a PIL Image
    img = Image.fromarray(image_stack[i])
    images.append(img)

# save first image and append with subsequent images
# duration is for each frame in ms
images[0].save('output/oct10measurement24.gif', 
    save_all=True, append_images=images[1:], duration=200, loop=0)
