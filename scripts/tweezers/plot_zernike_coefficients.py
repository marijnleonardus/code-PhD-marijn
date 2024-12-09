import numpy as np
import matplotlib.pyplot as plt

indices = np.arange(3, 15, 1)
coefficients = np.zeros(len(indices))
coefficients[0] = -1.84
coefficients[1] = -0.416
coefficients[2] = 0.461
coefficients[3] = 0.807
coefficients[4] = 0.84
coefficients[5] = 0.99
coefficients[6] = -0.15
coefficients[7] = -0.77
coefficients[8] = -0.45
coefficients[9] = 0.05
coefficients[10:] = 0.01

print(coefficients)

fig, ax = plt.subplots()
ax.grid()
ax.bar(indices, coefficients)
ax.set_xlabel(r'Zernike coefficient index $j$')
ax.set_ylabel(r'Value coefficient')

plt.show()
