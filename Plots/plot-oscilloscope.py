# athor: Marijn L. Venderbosch
# september 2022

# %% imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% loading data

file_location = '//physstor/cqt-common/KAT1/Data/2022-09-19/'

df = pd.read_csv(file_location + 'artiq_pulse_speed.txt',
                 sep='\t',
                 skiprows=[0])

# convert np array
array = np.array(df)

# obtain time and values. Multiply time by 0.5 because of error in export
# not sure why
time = array[:, 0] * 0.5
values = array[:, 1]


def rescale_data(input_array, averaging_window):
    begin_value = np.mean(input_array[0:averaging_window])
    end_value = np.mean(input_array[-averaging_window])
    shifted_array = values - end_value
    rescaled_array = 1 / (begin_value - end_value) * shifted_array
    return rescaled_array


values_rescaled = rescale_data(values, 50)
# %% plotting

plt.grid()
plt.plot(time, values_rescaled)
plt.xlabel('time')
plt.ylabel('voltage scaled')
plt.title('oscilloscope reading')
plt.axhline(0.5, color='red')
plt.show()
