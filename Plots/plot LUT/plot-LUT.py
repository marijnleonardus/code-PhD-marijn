#%% imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% load lut

def load_lut(file_location, file_name):
    """
    Loads .lut file which is separated by tabs, new lines separated by enters
    Subsequently saves first and second column of the csv file
    """
    pd_file = pd.read_csv(file_location + '/' + file_name, sep = '\t', header = None)
    np_array= np.array(pd_file)
    
    grey_values = np_array[:, 0]
    LUT_values = np_array[:, 1]
    
    return grey_values, LUT_values

grey_values, LUT_values = load_lut('LUT_files','slm6486_at813_scaled.lut')

#%% plotting

fig, ax = plt.subplots()
ax.plot(grey_values, LUT_values)

ax.set_xlabel('grey value ( 8-bit)')
ax.set_ylabel('LUT value (10-bit)')
