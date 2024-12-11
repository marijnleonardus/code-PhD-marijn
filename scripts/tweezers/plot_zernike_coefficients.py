import numpy as np
import matplotlib.pyplot as plt

# we use the Noll index convention. We use the first 15 polynomials. 
indices = np.arange(4, 16, 1)
coefficients = np.zeros(len(indices))
coefficients[0] = -1.385
coefficients[1] = -0.705
coefficients[2] = 0.299
coefficients[3] = 0.688
coefficients[4] = 1.19
coefficients[5] = 0.80
coefficients[6] = -0.45
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