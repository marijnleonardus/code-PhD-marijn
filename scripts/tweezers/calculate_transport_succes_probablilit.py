import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the image
data = np.round(np.array([
    [0.767, 0.926, 0.872, 0.968, 0.846],
    [0.923, 0.881, 0.808, 0.929, 0.963],
    [0.97, 0.877, 0.868, 0.816, 0.914],
    [0.9, 0.82, 0.941, 0.918, 0.9]
]), 2)

fig, ax = plt.subplots(figsize=(4, 3.5))

# Create the heatmap
# vmin and vmax match the colorbar in your image
im = ax.imshow(data, cmap='viridis', vmax=1.0)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set axis labels
ax.set_xlabel('Tweezer index, horizontal')
ax.set_ylabel('Tweezer index, vertical')

# Add text annotations rounded to 2 digits
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        ax.text(j, i, f'{val:.2f}', 
                ha="center", va="center", color="white")
        
# Set ticks to integer positions only
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))

# Disable the tick marks (the little lines) but keep the text labels
ax.tick_params(axis='both', which='both', length=0)

# Adjust ticks to match the image style
#ax.set_xticks(np.arange(data.shape[1]))
#ax.set_yticks(np.arange(-0.5, 4.0, 0.5)) # Match the 0.5 increment ticks

plt.tight_layout()
plt.show()