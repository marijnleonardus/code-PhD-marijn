# author: Marijn Venderbosch
# August 2024

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

# %% parameters and import data

# parameters
sample_rate = 1.0 # [Hz]

# import data. second column is repetition rate, first column time
data_location = 'T:\\KAT1\\Marijn\\FC1500measurements\\cavity drift\\'
data = pd.read_csv(data_location + 'august9result.dat', comment='#', sep='\s+')
data.columns = ['time', 'repetition_rate', 'carrier_envelope_offset']

# get repetition rate 1d array
rep_rate_array = data['repetition_rate'].to_numpy()

# %% compute allan dev 

FrequencyAnalysis = AllanDevFromDataset(rep_rate_array, sample_rate)

# plot frequency deviation (relative)
fig0, ax0 = plt.subplots()
ax0.plot(FrequencyAnalysis.compute_frac_freqs(), label='x')
ax0.set_xlabel('time [s]')
ax0.set_ylabel('relative frequency deviation')
ax0.legend()

m_list, allan_var = FrequencyAnalysis.compute_allan_var()
fig1, ax1 = plt.subplots()
ax1.plot(m_list, np.sqrt(allan_var), label='x')
ax1.set_xlabel('integration time [s]')
ax1.set_ylabel('allan deviation [Hz]')
ax1.legend()

plt.show()

# %%
