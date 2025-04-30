import numpy as np 
from scipy.constants import proton_mass, hbar, pi

kHz = 1e3
MHz = 1e6
h=hbar*2*pi
m=88*proton_mass

depth = 2.2*MHz*h
trap_freq_radial = 53*kHz
trap_freq_axial = 5.5*kHz


def calc_waist(depth, trap_freq_rad):
    waist0 = 1/(2*pi)*np.sqrt(4*depth/(m*trap_freq_rad**2))
    return waist0


def calc_rayleigh_length(depth, trap_freq_ax):
    rayleigh_length = 1/(2*pi)*np.sqrt(2*depth/(m*trap_freq_ax**2))
    return rayleigh_length


print(calc_rayleigh_length(depth, trap_freq_axial))
