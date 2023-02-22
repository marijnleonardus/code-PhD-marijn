# author: Marijn Venderbosch
# January 2023

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

picoscope_data = np.genfromtxt('plots/phase noise/679combbeat2.csv', delimiter=',', )

frequency_axis = picoscope_data[:, 0]
noise_dbm = picoscope_data[:, 1]
noise_dbc = noise_dbm - np.max(noise_dbm)

plt.plot(frequency_axis, noise_dbc)

