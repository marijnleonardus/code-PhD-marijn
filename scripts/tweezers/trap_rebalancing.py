import numpy as np
import io
import matplotlib.pyplot as plt

from modules.image_analysis_class import ImageStats
from utils.units import MHz

# raw data as a multi-line string
raw_data = """
4627032.245884439	4589964.515676846	4585161.677710683	4539801.178081187	4563018.319150109
4573019.232188906	4517739.410633722	4528842.958267027	4540973.690944418	4564278.633072353
4580396.8686715	4634138.525449547	4637448.047667573	4614455.3803761145	4543318.524528461
4555026.603303095	4554925.164582677	4612044.566166543	4604159.160614684	4637702.883860345
4653285.794298417	4615665.045248492	4601960.164568459	4610656.327524389	4552023.5669511175
"""

# Convert string to a NumPy array
matrix = np.loadtxt(io.StringIO(raw_data.strip()))
processed_matrix = np.round(matrix/MHz, 2)

# Output with ',' as the delimiter
# This prints the array formatted with commas
print(np.array2string(processed_matrix, separator=', '))

uniformity = ImageStats.calculate_uniformity(matrix)
print(f"Uniformity: {uniformity:.4f}")
plt.imshow(matrix, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('detuning (MHz)', rotation=270, labelpad=15)
plt.show()

# calculate standard deviation with Bessel's correction (ddof=1)
mean = np.mean(matrix)
stddev = np.std(matrix, ddof=1)
print(f"Relative Standard Deviation: {stddev/mean*100:.3f}", " %")
