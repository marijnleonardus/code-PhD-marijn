import numpy as np
import matplotlib.pyplot as plt

# we use the Noll index convention. We use the first 15 polynomials. 
indices = np.arange(5, 16, 1)
coefficients = np.zeros(len(indices))
coefficients[0] = 1.25
coefficients[1] = 0.04
coefficients[2] = -0.017
coefficients[3] = -0.15
coefficients[4] = 0.08
coefficients[5] = 0.726
coefficients[6] = -0.072
coefficients[7] = -0.227
coefficients[8] = 0.074
coefficients[9] = -0.089
coefficients[10:] = 0.13

print(coefficients)

fig, ax = plt.subplots(figsize=(4, 2.5))
ax.grid()
ax.bar(indices, coefficients)
ax.set_xlabel(r'Zernike coefficient $z_j$')
ax.set_ylabel(r'Value coefficient $c_j$')
fig.tight_layout()
#plt.show()
plt.savefig('scripts/optics/zernike-coefs.pdf')