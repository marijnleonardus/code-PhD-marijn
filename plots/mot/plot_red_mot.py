import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_location = r'S:\KAT1\Images\Artiq\SrMachine\2023\2023-11-08\20196'
file_name = r'\0002.tif'
print(file_location + file_name)
image = mpimg.imread(file_location + file_name)
plt.imshow(image, cmap = 'Reds')
plt.axis('off')
plt.show()