# author: Marijn Venderbosch
# november 2022

import numpy as np
import matplotlib.pyplot as plt

years = np.linspace(2010, 2022, 13)
publications = [343, 403, 463, 402, 457, 485, 553, 600, 762, 1122, 1362, 1842, 1754]

plt.plot(years, publications)
plt.show()