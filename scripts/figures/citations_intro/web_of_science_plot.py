# author: Marijn Venderbosch
# november 2022

# %% imports

import numpy as np
import matplotlib.pyplot as plt

# %% raw data

start_year = 2010
end_year = 2023
nr_years = end_year - start_year + 1
years = np.linspace(start_year, end_year, nr_years)
publications = [373, 458, 517, 475, 529, 602, 708, 828, 1066, 1543, 1937, 2578, 3251, 4265]

# %% plotting

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(years, publications)

ax.set_xlabel('Year')
ax.set_ylabel('Number of publications')

plt.savefig('citations_per_year.pdf',
            bbox_inches='tight',
            dpi=300)
plt.show()
