import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# set locations and file names, so that you can easily switch between
file_location_bb = r'T:KAT1\Marijn\redmot\redmot_bb'
file_location_sf = r'T:KAT1\Marijn\redmot\redmot_sf'

file_name_bb = r'\0002.tif'
file_name_sf = r'\sf_highpower.tif'

# choose one of the previously defined file_locations / names 
file_location = file_location_sf
file_name = file_name_sf

image = mpimg.imread(file_location + file_name)
plt.imshow(image, cmap = 'Reds')
plt.axis('off')
plt.show()