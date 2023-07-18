# author: Marijn Venderbosch
# november 2022

# %% imports

import numpy as np
import matplotlib.pyplot as plt

# %% raw data

years = np.linspace(2010, 2021, 12)
publications = [343, 403, 463, 402, 457, 485, 553, 600, 762, 1122, 1362, 1842]

# %% plotting

fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(years, publications)

ax.set_xlabel('Year')
ax.set_ylabel('Number of publications')

plt.savefig('citations_per_year.pdf',
            bbox_inches='tight',
            dpi=300)
plt.show()
