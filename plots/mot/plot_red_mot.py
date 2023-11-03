import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_location = 'T:\\KAT1\\Marijn\\redmot\\'
file_name = 'fluorescence.bmp'

image = mpimg.imread(file_location + file_name)
plt.imshow(image, cmap = 'Reds')
plt.axis('off')
plt.show()