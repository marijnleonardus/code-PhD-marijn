# author: Marijn Venderbosch
# november 2022

# %% imports

# packages
import numpy as np
import matplotlib.pyplot as plt

# user-defined
import sys
new_path = '/media/marijn/HDD/GitHub/code-PhD-marijn/general/'
sys.path.append(new_path)
from saving_plot import save_figure 

# %% raw data

years = np.linspace(2010, 2021, 12)
publications = [343, 403, 463, 402, 457, 485, 553, 600, 762, 1122, 1362, 1842]

# %% plotting

fig, ax = plt.subplots(figsize=(4,3))
ax.bar(years, publications)

ax.set_xlabel('Year')
ax.set_ylabel('Number of publications')

save_figure('publications_vs_year.pdf')
