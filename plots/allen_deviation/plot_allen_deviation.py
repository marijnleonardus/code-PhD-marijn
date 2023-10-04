# author: Yuri van der Werf, Marijn Venderbosch
# october 2023

import allantools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parameters
f0 = 194.4e12  # approximate ORS (optical reference system) refrequency [Hz]
drift_rate = 0.09369  # cavity drift as measured by beating the ORS with RF locked comb [Hz/s]
sample_rate = 1.0  # cavity freq. sampling rate [Hz]

# import data
file_location = 'T:\\KAT1\\Marijn\\'
file_name = 'cavity_drift_originexport.csv'
file_import_string = file_location + file_name
df = pd.read_csv(file_import_string, delimiter=',')

# obtain time and frequencies columns and convert to np array
# we import the beat frequencies but we need the actual frequencies
# so add f0
beat_frequencies = df['Comb-ORS beat'].to_numpy()
times = df['time'].to_numpy()
ors_freq = beat_frequencies + f0

# correct for cavity drift
ors_freq_dedrift = ors_freq + times*drift_rate

def compute_frac_freqs(freq_list):
    """convert frequency value list to fractional frequency values

    Args:
        freq_list (np array of floats): list of frequencies in Hz

    Returns:
        frac_freq_listype (np array of floats): list of fractional freqs. 
    """
    mean_value = np.mean(freq_list)
    freq_differences = freq_list - mean_value
    frac_freqs_list = freq_differences/mean_value
    return frac_freqs_list


# Compute a deviation using the Dataset class
def compute_allan_dev(freqs, sampling_rate):
    """compute allen devation using allantools library

    Args:
        freqs (np array floats): list of fractional frequencies 
        sampling_rate (float): sampling rate of data in [Hz]

    Returns:
        dataset: computed overlapping allen deviation
    """
    dataset = allantools.Dataset(data = freqs, data_type='freq', rate = sampling_rate)
    dataset.compute("oadev")
    return dataset

def calculate_allan_deviation_yuri(frequencies):
    """calculate allen deviation from definition used in 
    https://ww1.microchip.com/downloads/aemDocuments/documents/VOP/ApplicationNotes/
    ApplicationNotes/Oscillator+Short+Term+Stability+and+Allan+Deviation.pdf

    Args:
        frequencies (np array of floats): list of frequencies

    Returns:
        m_list (list of ints): list of m values
        allen_dev (np array of floats): list of allen deviations
    """

    allan_variance = []
    m_list = []
    new_m = 1

    # generate list of m, which are the number of samples to average over for each step
    while new_m < len(frequencies)/3:
        m_list.append(new_m)
        new_m = 3*new_m

    # for each m, compute allen variance
    for m in m_list:
        freq_list = np.array([np.average(frequencies[i*m : (i + 1)*m]) for i in range(int(len(frequencies)/m))])
        sum_term = 1/(2*len(freq_list) - 1)*np.sum((freq_list[1:] - freq_list[:-1])**2)
        allan_variance.append(sum_term)

    # compute allen deviation from allen variance
    allan_variance = np.array(allan_variance)
    return (m_list, allan_variance)



# convert to fractional frequencies, which is needed for the allentools library functions
ors_freqs_frac = compute_frac_freqs(ors_freq)
ors_freqs_frac_dedrift = compute_frac_freqs(ors_freq_dedrift)

# compute allen deviations for both data sets
m_list, allen_var_uncorrected = calculate_allan_deviation_yuri(ors_freqs_frac)
m_list, allen_var_dedrift = calculate_allan_deviation_yuri(ors_freqs_frac_dedrift)

fig, ax = plt.subplots()
ax.loglog(m_list, np.sqrt(allen_var_uncorrected))
ax.loglog(m_list, np.sqrt(allen_var_dedrift))

# write results to file
#dataset_uncorrected.write_results("output.dat")
#dataset_dedrift.write_results("output_dedrift.dat")


# Plot it using the Plot clas
# additional keyword arguments are passed to
# matplotlib.pyplot.plot()s
#plot = allantools.Plot()

#plot.plot(dataset_uncorrected, errorbars=True, grid=True)
#plot.plot(dataset_dedrift, errorbars=True, grid=True)

# You can override defaults before "show" if needed
#plot.ax.set_xlabel("Tau (s)")
#plot.ax.set_ylabel("Allan Deviation [Hz]")

plt.show()
