# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:52:09 2023

@author: Robert de keijzer
"""

"""script for fitting experimental data for radial dipole matrix elements
this is for 3P1 - 3S1, which are equivalent to 3P0 - 3S1 up to a Clebsch Gordan coefficient

I copied this script from Robert de keijzer"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from classes.unit_conversion.conversion_functions import *


def fit_func_rdme(x, a1, a2, b1, b2):
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2*x**2)

#Data from  Canzhu Tan et al 2022 Chinese Phys. Lett. 39 093202
#These are RDME values between 5s5p3P1 and 5s5n3S1
Nvalues=np.arange(19,41)
RDMEvalues=np.array([0.0281749, 0.0256654, 0.0234981, 0.0214449, 0.0198479, 0.018365,
0.0171103, 0.0160266, 0.015, 0.0140875, 0.013289, 0.0125475, 0.0118631, 0.0112357,
0.0106654, 0.0100951, 0.00969582, 0.00918251, 0.00872624, 0.00838403, 0.00804183, 0.00769962])

plt.scatter(Nvalues, RDMEvalues, label='data')

popt_rdme, pcov_rdme = curve_fit(fit_func_rdme, Nvalues, RDMEvalues, [0.05,0.06,0.05,0.005])
Nvalues=np.arange(19,70)
plt.plot(Nvalues, fit_func_rdme(Nvalues, *popt_rdme), 'g--',
         label='fit: a1=%5.3f, a2=%5.3f, b1=%5.3f, b2=%5.3f' % tuple(popt_rdme))
plt.xlim(18,70)
plt.show()


# Data from Phys. Rev. A 99, 022503
Nvalues2=np.arange(13,51)
energies=np.array([1341500517, 1347874127, 1352673833, 1356377995, 1359296416,
          1361636650, 1363541952, 1365113813, 1366425741, 1367532054, 
          1368473584, 1369281502, 1369979949, 1370587852, 1371120204, 
          1371589028, 1372004044, 1372373187, 1372702970, 1372998803, 
          1373265188, 1373505903, 1373724155, 1373922642, 1374103691, 
          1374269691, 1374421124, 1374560698, 1374689300, 1374808037, 
          1374917901, 1375019753, 1375114353, 1375202375, 1375284413, 
          1375360997, 1375432602, 1375499653],dtype=float)*10**6*2*np.pi

energiescm=freq_to_cmmin1(energies)

ionization_energy=45932.2036
def fit_func_energy(x, a1, b1, a2, b2, a3, b3):
    return ionization_energy-a1 * np.exp(-b1 * x) - a2 * np.exp(-b2*x**2)- a3 * np.exp(-b3*x**3)

plt.scatter(Nvalues2, energiescm, label='data')

popt_energy, pcov_energy = curve_fit(fit_func_energy, Nvalues2, energiescm, [2764.33,0.0784766,4914.71,0.0146274,243.563,0.0000301007])
Nvalues=np.arange(13,80)
plt.plot(Nvalues, fit_func_energy(Nvalues, *popt_energy), 'g--',
         label='fit: a1=%5.3f, b1=%5.3f, a2=%5.3f, b2=%5.3f, a3=%5.3f, b3=%5.3f' % tuple(popt_energy))
plt.xlim(10,80)
plt.show()

energy_5s5p3P1=14504.334
def fitted_rdme_energy_3P1(N):
    return [fit_func_rdme(N, *popt_rdme),fit_func_energy(N, *popt_energy)-energy_5s5p3P1]


#To estimate RDME values of 5s5p3P0 and 5s5n3S1 we use the Wigner 6J relations
# as in Eq. (4) of https://www.phys.ksu.edu/reu2015/danielkeylon/Wigner.pdf
# we find that they are a factor 1/Sqrt[3] lower for 3P0 than for 3P1 so we get
energy_5s5p3P0=14317.507
def fitted_rdme_energy_3P0(N):
    return [fit_func_rdme(N, *popt_rdme)*np.sqrt(1/3),fit_func_energy(N, *popt_energy)-energy_5s5p3P0]

#For 5s5p3P2 we can do the same to find Sqrt[5/3]
energy_5s5p3P2=14898.545
def fitted_rdme_energy_3P2(N):
    return [fit_func_rdme(N, *popt_rdme)*np.sqrt(5/3),fit_func_energy(N, *popt_energy)-energy_5s5p3P2]

#From https://arxiv.org/pdf/2001.04455.pdf we estimate for n=61 a decay rate of 64 Hz
# our predictions here give 57 Hz which is not bad