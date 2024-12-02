# author: Marijn Venderbosch
# december 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parameters
data_location = 'T:\\KAT1\\Marijn\\FC1500measurements\\rep_rate_logging\\white_rabbit\\'

# import data as dataframe

df = pd.read_csv(data_location + 'allan_dev_dec2.dat', skiprows=4, sep="\t", names=["Gate Time", "Deviation"])

integration_times = df['Gate Time'].to_numpy()
deviations = df['Deviation'].to_numpy()

fig, ax = plt.subplots()
ax.grid()
ax.scatter(integration_times, deviations)
ax.set_xscale('log')
ax.set_xlabel('Gate time [s]')
ax.set_ylabel('Allan deviation')
ax.set_yscale('log')
plt.show()
