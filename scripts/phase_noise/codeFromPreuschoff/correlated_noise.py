"""code from the thesis of Preuschoff (2023
https://tuprints.ulb.tu-darmstadt.de/23242/"""


import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def read_data(filename):
    data = pd.read_csv(filename, delimiter=",", names=['time','signal1','signal2','signal3'])
    x = data['time'].astype(float)
    y = data['signal1'].astype(float)
    return np.array(x),np.array(y)


def sample_phase_noise(N, sampling_rate, noise_model):
    """
    Returns N samples of colored noise generated from a noise model.
        N: Number of samples
        sampling_rate: set the sampling rate, such that the Nyquist frequency is well
            above the noise psd cut-off
        noise_model: continuous model for sqrt(psd) [rad/sqrt(Hz)] (e.g. use a cubic
            spline generated from measured data)
    """
   
    white_psd = np.fft.rfft(np.random.randn(N)) # FFT of uncorrelated sample
    freq = np.fft.rfftfreq(N, d = 1/sampling_rate) # Fourier frequencies of the sample
    weights = noise_model(freq) # Generate weights from the noise model avoid f=0 for f**-x models
    # Normalize weighting function: ||noise_model||_2 / sqrt(mean_psd)
    # Attention: f**-x models are not integrable. Avoid freq[0] = 0 and rescale empirically.
    weights = weights/np.sqrt(np.mean(weights**2))*np.sqrt(np.trapz(noise_model(freq)**2,freq))
    shaped_psd = white_psd*weights
    t=np.arange(N)*1/sampling_rate
    shaped_noise = np.fft.irfft(shaped_psd) # obtain samples by inverse Fourier transform
    return t, np.real(shaped_noise)

### Get noise_model from measured data
f, noise = read_data('noise_models/phasenoise_Raman.csv')
sampling_rate = np.max(f)*2
noise_model = CubicSpline(f,np.sqrt(noise))

### Sample random noise
t, phase_noise = sample_phase_noise(int(1e6), sampling_rate, noise_model)


