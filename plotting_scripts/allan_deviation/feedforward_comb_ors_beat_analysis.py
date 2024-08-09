# author: Marijn Venderbosch
# october 2023

# %% imports 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# append path with 'modules' dir in parent folder
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
modules_dir = os.path.abspath(os.path.join(script_dir, '../../modules'))
sys.path.append(modules_dir)

# user defined functions
from frequency_analysis_class import AllanDevFromDataset

# %% general parameters

# parameters
f0 = 194.4e12  # approximate ORS (optical reference system) refrequency [Hz]
sample_rate = 1.0  # cavity freq. sampling rate [Hz]
data_location = 'T:\\KAT1\\Marijn\\FC1500measurements\\cavity drift\\'

# %% holdover mode

# cavity drift rates as obtained from origin linear fit of comb-ORS beat
drift_rate_holdover = 0.09369  

# import data as dataframe
df_holdover = pd.read_csv(data_location + 'cavity_drift_originexport.csv', delimiter=',')

# obtain time and frequencies columns and convert to np array
# we import the beat frequencies but we need the actual frequencies, so add the ORS freq. f0
beat_freqs_holdover = df_holdover['Comb-ORS beat'].to_numpy()
times_holdover = df_holdover['time'].to_numpy()
ors_freq_holdover = beat_freqs_holdover + f0

# correct for cavity drift
ors_freq_holdover_dedrift = ors_freq_holdover + times_holdover*drift_rate_holdover

# %% GPS disciplined mode

# cavity drift as measured by GPS disciplined ocillator comb lock [Hz/s]
drift_rate_gps = 0.08919  

# import data as dataframe
df_gps = pd.read_csv(data_location + 'cavity_drift_originexport.csv', delimiter=',')

# obtain data from import and skip first 18000 data points because something wrong with data there
beat_freqs_gps = df_gps['Comb-ORS beat'].to_numpy()
beat_freqs_gps = beat_freqs_gps[18000:]
times_gps = df_gps['time'].to_numpy()
times_gps = times_gps[18000:]
ors_freq_gps = beat_freqs_gps + f0

# correct for cavity drift
ors_freq_gps_dedrift = ors_freq_gps + times_gps*drift_rate_gps

# %% Plot fractional frequncies; freq. stability

# Create objects for all 4 datasets
FreqsHoldover = AllanDevFromDataset(ors_freq_holdover, sample_rate)
FreqsHoldoverDedrift = AllanDevFromDataset(ors_freq_holdover_dedrift, sample_rate)
FreqsGPS = AllanDevFromDataset(ors_freq_gps, sample_rate)
FreqsGPSDedrift = AllanDevFromDataset(ors_freq_gps_dedrift, sample_rate)

fig0, ax0 = plt.subplots()
ax0.plot(FreqsHoldover.compute_frac_freqs(), label='holdover')
ax0.plot(FreqsHoldoverDedrift.compute_frac_freqs(), label='holdover, dedrift corrected')
ax0.plot(FreqsGPS.compute_frac_freqs(), label='gps disciplined')
ax0.plot(FreqsGPSDedrift.compute_frac_freqs(), label='gps disciplined, dedrift corrected')
ax0.legend()
#plt.show()

# %% plot allan deviation

m_list_holdover, allan_var_holdover = FreqsHoldover.compute_allan_var()
m_list_holdover, allan_var_holdover_dedrift = FreqsHoldoverDedrift.compute_allan_var()

m_list_gps, allan_var_gps_ = FreqsGPS.compute_allan_var()
m_list_gps, allan_var_gps_dedrift = FreqsGPSDedrift.compute_allan_var()

fig1, ax1 = plt.subplots()
ax1.loglog(m_list_holdover, np.sqrt(allan_var_holdover), 'b--', label='mode: holdover')
ax1.loglog(m_list_holdover, np.sqrt(allan_var_holdover_dedrift), 'b', label='mode: holdover, dedrift corrected')
ax1.loglog(m_list_gps, np.sqrt(allan_var_gps_), 'r--', label='mode: gps-disciplined')
ax1.loglog(m_list_gps, np.sqrt(allan_var_gps_dedrift), 'r', label='mode: gps-disciplined, dedrift corrected')

ax1.legend()
plt.show()
