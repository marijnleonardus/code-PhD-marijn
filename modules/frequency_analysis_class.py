# author: Yuri van der Werf, Marijn Venderbosch
# August 2024

import numpy as np
import allantools


class AllanDevFromDataset:
    def __init__(self, dataset, sampling_rate: float):
        self.freqs = dataset
        self.sample_rate = sampling_rate

    def compute_frac_freqs(self):
        """convert frequency value list to fractional frequency values

        Args:
            freq_list (np array of floats): list of frequencies in Hz

        Returns:
            frac_freq_listype (np array of floats): list of fractional freqs. 
        """
        mean_value = np.mean(self.freqs)
        freq_differences = self.freqs - mean_value
        frac_freqs_list = freq_differences/mean_value
        return frac_freqs_list

    def compute_allan_var_allantools(self):
        """compute allen devation using allantools library

        Args:
            freqs (np array floats): list of fractional frequencies 
            sampling_rate (float): sampling rate of data in [Hz]

        Returns:
            dataset: computed overlapping allen deviation
        """
        # compute fractional frequencies
        frac_freqs = self.compute_frac_freqs()

        result = allantools.Dataset(data=frac_freqs, data_type='freq', rate=self.sample_rate)
        result.compute("oadev")
        return result

    def compute_allan_var(self):
        """calculate allen deviation from definition used in 
        https://ww1.microchip.com/downloads/aemDocuments/documents/VOP/ApplicationNotes/
        ApplicationNotes/Oscillator+Short+Term+Stability+and+Allan+Deviation.pdf
        this function was written by Yuri van der Werf

        Args:
            frequencies (np array of floats): list of (fractional) frequencies

        Returns:
            m_list (list of ints): list of m values
            allen_dev (np array of floats): list of allen deviations
        """

        # compute fractional frequencies
        frac_freqs = self.compute_frac_freqs()

        allan_variance = []
        m_list = []
        new_m = 1

        # generate list of m, which are the number of samples to average over for each step
        while new_m < len(frac_freqs)/2:
            m_list.append(new_m)
            new_m = 2*new_m

        # for each m, compute allen variance
        for m in m_list:
            freq_list = np.array([np.average(frac_freqs[i*m : (i + 1)*m]) for i in range(int(len(frac_freqs)/m))])
            sum_term = 1/(2*len(freq_list) - 1)*np.sum((freq_list[1:] - freq_list[:-1])**2)
            allan_variance.append(sum_term)

        # compute allen deviation from allen variance
        allan_variance = np.array(allan_variance)
        return (m_list, allan_variance)
