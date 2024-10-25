# author: Marijn Venderbosch
# January 2023

import numpy as np
import pandas as pd
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# %% data and variables

df = pd.read_csv('data/frequencynoise.csv', delimiter=';', decimal=',', header=None)
picoscope_data = np.array(df)

pi = np.pi

omega = picoscope_data[:, 0]*2*pi
noise = picoscope_data[:, 1]

# rabi freqs of interest for us
rabi_min = 2*pi*0.3e6  # [Hz]
rabi_max = 2*pi*30e6  # [Hz]
rabifreqs = np.logspace(np.log10(rabi_min), np.log10(rabi_max), 20)

# %% functions

class IntegratePhaseNoise:
    
    def window_function_1(r, w, t):
        """
        windows function that selects relevant of frequency spectrum as a function of time
        see thesis Anant Kale, p. 23
    
        Parameters
        ----------
        r : float
            rabi frequency in [Hz].
        w : float
            frequency, independent variable (omega) [Hz].
        t : float
            time in [s].
    
        Returns
        -------
        window : array
            spectral window function that selects parts of spectrum.
    
        """
        
        # compute numerator
        term1=3*r**2+w**2+(r**2-w**2)*np.cos(2*r*t)
        term2=-4*r**2*np.cos(r*t)*np.cos(w*t)-4*r*w*np.sin(r*t)*np.sin(w*t)
        numerator = term1+term2
        
        # compute denominator and fraction
        denominator = (r**2-w**2)**2*t**2
        window = numerator/denominator
        return window 
    
    
    def product_function_1(rabi, omega):
        """
        computes product of window function with frequency noise spectrum
        so that this function can be integrated at time t1=pi/rabi
    
        Parameters
        ----------
        rabi : float
            rabi frequency in [Hz].
        omega : float
            frequency in [Hz].
    
        Returns
        -------
        product : array
            window function times noise spectrum
    
        """
        t1 = pi/rabi
        window1t = IntegratePhaseNoise.window_function_1(rabi, omega, t1)
        noise_spectrum = noise
        
        product = window1t * noise_spectrum
        return product
    
    
    def compute_fidelity_error(rabi):
        """
        computes fidelity error as result of phase noise 
        see thesis Anant Kale eq. 4.24 and eq. 4.57
    
        Parameters
        ----------
        rabi : float
            rabi frequency in [Hz].
    
        Returns
        -------
        fidelity_error : float
            error in pi pulse fidelity.
    
        """
        # enchance rabi freq by factor sqrt(2) because of blockade
        rabi = np.sqrt(2)*rabi
        
        # compute product function for specific rabi frequency and frequencies array
        product1t = IntegratePhaseNoise.product_function_1(rabi, omega)
        
        # integrate this spectrum
        integral = simpson(product1t, omega)
        
        fidelity_error = pi**2/(16*rabi**2)*integral
        return fidelity_error

 
# %% compute fidelity error as a function of rabi frequency

fidelities = []

for rabifreq in rabifreqs:
    error = IntegratePhaseNoise.compute_fidelity_error(rabifreq)
    fidelities.append(error)

fidelities = np.array(fidelities)

# %% plotting

# plot frequency noise spectrum
fig, ax = plt.subplots()
ax.plot(omega, noise, label=r'Noise spectrum $S_{\nu}(\omega)$')
ax.set_xlabel('Fourier frequence [Hz]')
ax.set_ylabel(r'$S_{\nu}$ Hz^2/Hz')

# plot noise multiplied with window function for example rabi freq of 5 MHz
product1t = IntegratePhaseNoise.product_function_1(2*pi*5e6,omega)
ax.plot(omega, product1t, label=r'$S_{\nu}(\Omega)\cdot W_1(\Omega=2\pi*5$ MHz)')
ax.legend()

# plot fidelity error as a function of rabi freq
fig2, ax2 = plt.subplots()
ax2.scatter(rabifreqs/(2*pi*1e6), fidelities)
ax2.set_yscale('log')
ax2.set_ylabel('$1-\mathcal{F}_2$')
ax2.set_xscale('log')
ax2.set_xlabel('')

