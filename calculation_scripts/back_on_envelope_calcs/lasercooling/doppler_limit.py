# calculate doppler limit (for broadened linewidth as well)

from scipy.constants import hbar, Boltzmann
import numpy as np

linewidth = 7.4*1e3  # Rad/s
saturation = 75

gamma = 2*np.pi*linewidth  # Hz

t_doppler = hbar*gamma/(2*Boltzmann)*np.sqrt(1+saturation)
print(str(t_doppler*1e6) + ' uK')