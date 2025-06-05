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
from laser_class import AllanDevFromDataset
from data_handling_class import read_csv_file, pandas_read_datfile

# %% parameters and import data

# parameters
sample_rate = 1.0 # [Hz]
file_type = "csv"

# import data. second column is repetition rate, first column time
file_path = r"\\physstor\cqt-t\KAT1\Comb_measurements"
file_name = r"AU06792-log_reprate_beat.csv"

if file_type == "csv":
    # read the data using the csv function
    data = read_csv_file(file_path, file_name)

    # get repetition rate 1d array
    rep_rate_array = np.array([float(row[0]) for row in data if row[0].lower() != 'null'])

elif file_type == "dat":
    data = pandas_read_datfile(file_path, file_name)
    data.columns = ['time', 'repetition_rate', 'carrier_envelope_offset']

    # get repetition rate 1d array
    rep_rate_array = data['repetition_rate'].to_numpy()
else:
    raise "file type not supported"

# %% compute allan dev 

FrequencyAnalysis = AllanDevFromDataset(rep_rate_array, sample_rate)

# plot frequency deviation (relative)
fig0, ax0 = plt.subplots()
ax0.loglog(FrequencyAnalysis.compute_frac_freqs(), label='x')
ax0.set_xlabel('time [s]')
ax0.set_ylabel('relative frequency deviation')
#ax0.set_ylim(-1e-10, 0.5e-10)
ax0.legend()

m_list, allan_var = FrequencyAnalysis.compute_allan_var()
fig1, ax1 = plt.subplots()
ax1.loglog(m_list, np.sqrt(allan_var), label='x')
ax1.set_xlabel('integration time [s]')
ax1.set_ylabel('allan deviation [Hz]')
ax1.legend()

plt.show()

# %%
