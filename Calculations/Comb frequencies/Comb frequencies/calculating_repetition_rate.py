# Author: Marijn Venderbosch
# november 2022

"""script computes ideal repetition rate in order to lock n CW lasers to a single frequency comb

f_CW,n = 2 * f_CEO + m_n f_rep + f_beat,n

where:
    * f_CW_n is the beat of the CW laser with the comb
    * f_CEO is the carier envelope offset. Factor 2 because comb is freq. doubled
    * m_n is the mode number of CW laser n
    * f_rep is the repetition rate which is varied around 250 MHz
    
The script computes the beat frequencies f_beat,n for n CW lasers
and checks whether they are in the range that the DDS can lock to (25-50 MHz) 
"""

# %% imports

import numpy as np
from scipy.constants import c

# %% variables

# list of all CW frequencies, all units are in Hz
f1 = c / 922e-9  
f2 = c / 813e-9  
f3 = c / 707e-9
f4 = c / 698e-9
f5 = c / 689e-9
f6 = c / 688e-9
f7 = c / 679e-9

freq_CW_list = [f1, f2, f3, f4, f5, f6, f7]

# carrier envelope offset 
freq_ceo = 35e6  # Hz 

# DDS lock window
dds_lock_window = 60e6

# number of repetition rates to try
number_data_points = 1e4

# %% functions

def get_repetition_rate_array(number_data_points):
    
    """returns a matrix of repetition rates to scan over. Centered around 250 MHz
    Scan range of 1 MHz is used"""
    
    center = 250e6  # Hz
    scan_range = 1e6  # Hz
    number_steps = int(number_data_points + 1)
    
    repetition_rate_array = np.linspace(center - scan_range, center + scan_range, number_steps)
    return repetition_rate_array


def return_dds_limits(dds_lock_window):
    
    """ dds locking parameters. The DDS can lock in a limited window centered
    # around 35 MHz"""
    
    dds_lock_center = 35e6

    dds_lock_lower = dds_lock_center - 0.5 * dds_lock_window
    dds_lock_upper = dds_lock_center + 0.5 * dds_lock_window
    return dds_lock_lower, dds_lock_upper


def beat_cw_comb_positive(freq_n, freq_rep):
    
    """determine beat frequency of CW laser with optical frequency comb
    CEO is subtracted twice, because CW lasers lock to part spectrum that is 
    frequency doubled. This function works when f_beat > 0"""
    
    f_beat_positive = np.mod(freq_n - 2 * freq_ceo, freq_rep)
    return np.abs(f_beat_positive)


def beat_cw_comb_negative(freq_n, freq_rep):
    
    """similar to 'beat_CW_comb_positive' but for negative beat. ,
    This is equivalent to subtracting the beat frequency from the repetition rate"""
    
    f_beat = np.mod(freq_n - 2 * freq_ceo, freq_rep)
    f_beat_negative = freq_rep - f_beat
    return np.abs(f_beat_negative)


def squares_cost_function(input_array, ideal_value):
    
    """returns cost function, which is square of errors with respect to ideal value"""
    
    squares = (input_array - ideal_value)**2
    sum_squares = np.sum(squares)
    return sum_squares


# %% main sequence
   
def main():
    # get dds lock parameters and repetition rate values
    dds_lock_lower, dds_lock_upper = return_dds_limits(dds_lock_window)
    freq_rep_array = get_repetition_rate_array(number_data_points)
    
    # initialize empty list for storing the results from the repetition rate scan
    freq_rep_list = []
    minus_beat_CW_list = []
    plus_beat_CW_list = []
    
    # scan repetition rate
    for f_rep in freq_rep_array:
        
        # compute beat frequencies (positive and negative) for all CW lasers with the comb
        list_plus_beats = [beat_cw_comb_positive(freq_CW, f_rep) for freq_CW in freq_CW_list]
        list_minus_beats = [beat_cw_comb_negative(freq_CW, f_rep) for freq_CW in freq_CW_list]
        
        # check if beat frequencies are in lockable range of the DDS
        if max(list_plus_beats) < dds_lock_upper or max(list_minus_beats) < dds_lock_upper: 
            if min(list_plus_beats) > dds_lock_lower or min(list_minus_beats) > dds_lock_lower: 
                
                # store results to be used laser
                freq_rep_list.append(f_rep)
                plus_beat_CW_list.append(list_plus_beats)
                minus_beat_CW_list.append(list_minus_beats)
                
    # compute total error from ideal locking range 


    print("number of solutions: " + str(len(freq_rep_list)))
    
if __name__ == "__main__":
    main()