# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:17:00 2022

@author: Marijn
"""

# %% import oscilloscope class

from class_rigol_scope import RigolScope
import matplotlib.pyplot as plt


# %% variables

ip_address = '10.0.30.12'
channel_number = 4


# %% execute script

def main():
    scope1 = RigolScope(ip_address)
  
    time_matrix, spectrum_data_matrix = scope1.get_data(channel_number)
    
    fig, ax = plt.subplots()
    ax.plot(time_matrix, spectrum_data_matrix)


if __name__ == "__main__":
    main()
