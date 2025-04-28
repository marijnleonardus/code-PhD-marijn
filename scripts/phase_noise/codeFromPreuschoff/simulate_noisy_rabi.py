"""code from the thesis of Preuschoff (2023
https://tuprints.ulb.tu-darmstadt.de/23242/"""


from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy import signal
import time

def get_noise_model(filename, sampling_rate, noise_scale=1):
    ### Get noise model from measured data
    f, noise = read_data(filename)
    # extrapolate noise data
    if 1.1*np.max(f) < sampling_rate/2:
        f0 = np.logspace(np.log10(f[-1]*1.2), np.log10(sampling_rate/2), 100)
        f = np.append(f, f0)
        n0 = np.zeros(len(f0))+noise[-1]
        noise = np.append(noise, n0)
    noise = np.sqrt(noise_scale*noise)
    return CubicSpline(f, noise)

def simulate_phase_noise(omega_rabi, delta, noise_model, model_name=None,
                          rabi_cycles=10, rabi_res=100, avg=1000, save_results=True, weight_scale=1):
    """
    Set simulation parameters:
    omega_rabi: (angular) Rabi frequency
    delta: detuning from resonance (expectation value of the frequency distribution)
    noise_model: path to noise model csv-file. The model defines the phase noise PSD [rad**2/Hz].
    Data are interpolated -- make sure they cover the relevant frequency range. Extrapolation may lead to unexpected behaviour.
    noise_scale: scale noise with a factor, e.g. due to SHG or unit conversion
    rabi_cycles: run simulation over # rabi cycles
    avg: Average over # of realizations
    limit_bw: Set sampling rate, for low Rabi frequencies. If None, the sampling rate is chosen to cover the frequency range given by the noise model.
    random_delta: Include random frequency offset considered constant during the simulation time, e.g. caused by a finite laser linewidth.
    save_results: save simulation results as .csv
    """

    # calculate sampling rate and sample length
    sampling_rate = int(rabi_res*(omega_rabi*2/np.pi))
    N = rabi_res*rabi_cycles
    t = np.arange(N)*1/sampling_rate

    # Define Hamiltonian:
    g = Qobj([[1], [0]]) # start in the ground state
    e = Qobj([[0],[1]])
    e_proj = Qobj([[0,0],[0,1]])
    H0 = -delta/2*sigmaz() 
    H1 = omega_rabi/2*Qobj([[0,0],[1,0]])
    H2 = omega_rabi/2*Qobj([[0,1],[0,0]])

    output= []
    print('Start simulation: Omega_rabi = %.3f MHz'%(omega_rabi/(2*np.pi*1e6)))
    t0 = time.time()
    for i in range(avg):
        #Draw an instance of phase noise 
        phase_noise = sample_phase_noise(N,sampling_rate,noise_model,weight_scale)
        S1 = Cubic_Spline(t[0], t[-1],np.exp(1j*phase_noise))#interpolate discrete  random data
        S2 = Cubic_Spline(t[0], t[-1],np.exp(-1j*phase_noise))

        'solve Schroedinger equation'
        H = [H0,[H1,S1],[H2,S2]]
        result = sesolve(H,g,t,e_ops=[e_proj])
        output.append(result)

    'average over all realizations'
    mean_list = []
    for result in output:
        mean_list.append(np.abs(result.expect[0]))
        mean = np.mean(mean_list,axis=0)
    print('Done in %.0f s'%(time.time()-t0))

    'save data'
    if save_results:
        df = pd.DataFrame()
        df['Time'] = np.concatenate((np.array(['omega_rabi','delta','noise_model',' rabi_cycles','res_t']),t),axis=None)
        df['Mean'] = np.concatenate((np.array([omega_rabi,delta,model_name,rabi_cycles,rabi_res]),mean),axis=None)
        df.to_csv('results_phase_noise_rabi_%.3fMHz_%.0f_cycles.csv'%(omega_rabi/(2* np.pi*1e6),rabi_cycles))
    return t,mean,omega_rabi

omega_rabi = 10e6 
t_res = 100
noise_model = get_noise_model('phasenoise_Raman.csv', omega_rabi*t_res, noise_scale=1) 
t,mean,omega_rabi = simulate_phase_noise(2*np.pi*omega_rabi,0,noise_model, model_name = 'Raman phase noise' rabi_cycles=100, rabi_res = t_res, avg = 1000, save_results=True)