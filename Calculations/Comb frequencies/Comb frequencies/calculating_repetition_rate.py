# Author: Marijn Venderbosch
# november 2022

"""script computes ideal repetition rate in order to lock n CW lasers to a single frequency comb

f_CW,n = 2 * f_CEO + m_n f_rep + f_beat,n

where:
    * f_CW_n is the beat of the CW laser with the comb
    * f_CEO is the carier envelope offset. Factor 2 because comb is freq. doubled
    * m_n is the mode number of CW laser n
    * f_rep is the repetition rate which is varied around 250 MHz
    
The script computes the beat frequencies f_beat,n for n CW lasers for a variety of reptition rates.
It computes a cost function which is a sum of squares with the ideal beat frequencies the DDS can lock to,
which is by default 35 MHz.

For each repetition rate, it computes this cost. 
The solution with min. cost is chosen.
Subsequently, all CW beats for this deal repetition rate are printed. 
Also, it is shown whether this is a positive or negative beat.
"""

import numpy as np
from scipy.constants import c

# %% variables

# list of all CW frequencies, all units are in Hz

"""the blue 922/461 laser is excluded from the calculation for 2 reasons:
    1. It is frequency, not phase locked: limited precision
    2. Its DDS has a much larger allowed lock range."""
# f_922 = 325.25186 * 1e12

"""The 813 nm laser is excluded as well: this frequency is not linked to an atomc transition and does not 
need preceision in the MHz regime. We can freely move it with the DDS. """
# f_813 = c / 813.4274 nm 

# repump 3P2 -> 3S 1
# value from UvA collaborators, corrected for AOM upshift
f_707 = 423.91348 * 1e12 - 2 * 100e6

# clock transition
# https://doi.org/10.1038/srep17495
f_698 = 429.2280664180083 * 1e12

# red MOT transition
# https://doi.org/10.1103/PhysRevLett.91.243002 corrected for AOM upshift
f_689 = 434.829121311 * 1e12 - 2 * 100e6 

# repump 3P1 -> 3S1
f_688 = c / 688e-9

# repump 3P0 -> 3S1
# value from collaborators UvA, corrected for AOM upshift
f_679 = 441.33267 * 1e12 - 2 * 100e6

freq_CW_list = np.array([f_679, f_688, f_689, f_698, f_707])

# carrier envelope offset, and repetition rate center value (249-251 Mhz)
freq_ceo = 35e6  # Hz 
freq_repetition = 250e6   # Hz

# DDS lock window: DDS can lock in limited window. Gives size of this window. 
dds_lock_window = 30e6

# number of repetition rates to try
rep_rate_attemps = 2e7

# %% functions


def get_repetition_rate_array(number_rep_rate_attemps):
    
    """returns a matrix of repetition rates to scan over. 
    Centered around 250 MHz
    Scan range of 1 MHz is used: allowed rep. rate of Menlo is 249-251 MHz"""
    
    center = freq_repetition  # Hz
    scan_range = 1e6  # Hz
    number_steps = int(number_rep_rate_attemps + 1)
    
    repetition_rate_array = np.linspace(center - scan_range, center + scan_range, number_steps)
    return repetition_rate_array


def beat_cw_comb(freq_n, freq_rep):
    
    """
    f_CW,n = 2 * f_CEO + m_n f_rep + f_beat,n
    
    determine beat frequency of CW laser with optical frequency comb
    CEO is subtracted twice, because CW lasers lock to part spectrum that is 
    frequency doubled. This function works when f_beat > 0"""
    
    f_beat = np.mod(freq_n - freq_ceo, freq_rep)
    return np.abs(f_beat)


def square_cost_function(input_value, ideal_value):
    
    """returns cost function, which is square of errors with respect to ideal value"""
    
    squared = (input_value - ideal_value)**2
    return squared


# %% main sequence
   
def main():
    
    # get repetition rates to loop over
    freq_rep_array = get_repetition_rate_array(rep_rate_attemps)
    
    # initialize empty list for storing the results from the repetition rate scan
    freq_rep_list = []
    beat_cw_list = []
    
    # scan repetition rate
    for f_rep in freq_rep_array:
        
        # compute beat frequencies (positive and negative) for all CW lasers with the comb
        beats_array = np.array([beat_cw_comb(freq_CW, f_rep) for freq_CW in freq_CW_list])
        
        freq_rep_list.append(f_rep)
        beat_cw_list.append(beats_array)
    
        # # check if beat frequency is in lockable range of the DDS, 
        # # distinguishing between positive and negative beat. 
        # # Only store result if all CWs are in lockable range
        # if all(lim_down < beat < lim_up or (f_rep-lim_up) < beat < (f_rep-lim_down) for beat in beats_array):
        #     # store result in list
        #     
    
    costs_list = []
    
    # compute cost function for all CW beats, which is the square of the difference
    # with respect to the ideal DDS locking frequency of 35 MHz (+ or -) beat
    for beats_array in beat_cw_list:
        
        cost_unit = []
        for beat in beats_array:
            cost = min(square_cost_function(beat, freq_ceo),
                       square_cost_function(beat, freq_repetition - freq_ceo))
            cost_unit.append(cost)
        costs_list.append(sum(cost_unit))

    # return ideal solution and location in the list
    min_cost = min(costs_list)
    solution_index = costs_list.index(min_cost)
    
    # return ideal repetition rate and beat frequencies
    ideal_repetition_rate = freq_rep_list[solution_index]
    ideal_beat_frequencies = beat_cw_list[solution_index]
    
    # evaluate result, printing ideal repetition rate as well as all CW beats
    print("Optimum repetition rate: " + str(ideal_repetition_rate))
    print("CW beat frequencies: ")

    # list of CW frequencies rounded
    f_707_wl = "707 nm"
    f_698_wl = "698 nm"
    f_689_wl = "707 nm"
    f_688_wl = "698 nm"
    f_679_wl = "679 nm"
    cw_wavelengths = [f_707_wl, f_698_wl, f_689_wl, f_688_wl, f_679_wl]

    for i in range(len(cw_wavelengths)):
        
        # check if beat is + or -
        if ideal_beat_frequencies[i] < freq_repetition / 2:
            
            # beat is +, print (rounded) result
            print(cw_wavelengths[i] + " : " + str(+np.round(ideal_beat_frequencies[i] / 1e6,
                                                            decimals=2)) + " MHz")
        else:
            # beat is -, print difference rep. reate and beat freq. 
            print(cw_wavelengths[i] + " : " + 
                  str(-np.round((ideal_repetition_rate - ideal_beat_frequencies[i]) / 1e6,
                                decimals=2)) + " MHz (minus beat)")
            
    # return result so it is visible in variable explorer   
    return ideal_repetition_rate, ideal_beat_frequencies


if __name__ == "__main__":
    ideal_repetition_rate_comb, ideal_beat_frequencies_cw = main()
