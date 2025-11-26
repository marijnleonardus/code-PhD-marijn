# author: Yuri van der Werf, Marijn Venderbosch
# August 2024

import numpy as np
from scipy.constants import pi, hbar, c, epsilon_0, elementary_charge
import scipy.constants
import allantools
import os 
import sys

# add local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.abspath(os.path.join(script_dir, '../lib'))
if lib_dir not in sys.path:
    sys.path.append(lib_dir)
from setup_paths import add_local_paths
add_local_paths(__file__, ['../modules', '../utils'])

from units import MHz, um, mW, nm
from optics_class import GaussianBeam
from atom_class import Sr


class AllanDevFromDataset:
    def __init__(self, dataset: np.ndarray, sampling_rate: float):
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


class AtomLightInteraction:
    @staticmethod
    def calc_rydberg_rabi_freq(n, intensity: float, j_e: int):
        """computes Rabi frequency to the rydberg state
        given RDME (radial dipole matrix element)

        Args
            rdme (float): radial dipole matrix elment in [atomic units].
            intensity (float) laser intensity in [W/m^2].
            j_e (integer): quantum number J for rydberg state.

        Returns:
            rabi (float): rabi freq. in [rad/s]."""
        
        e0 = elementary_charge # C
        a0 = scipy.constants.physical_constants['Bohr radius'][0] # m

        # calculate radial dipole matrix elements
        rdme = Sr.get_rdme(n)

        # calculate angular rabi freq. from Madjarov thesis eq. (4.5)
        rydberg_rabi = (rdme*e0*a0)/hbar*np.sqrt(2*intensity/(epsilon_0*c*(2*j_e + 1)))
        return rydberg_rabi
    
    @staticmethod
    def calc_dc_stark_shift(polarizability: float, electric_field: float):
        """see paper Mohan 2022 for Sr88 data

        Args:
            polarizability (float): in units of [MHz cm^2 V^-2].
            electric_field (float): in units of [V/cm].

        Returns
            dc_stark_shift (float): DC Stark shift in Hz"""
        dc_stark_MHz = 1/2*polarizability*electric_field**2
        dc_stark_Hz = dc_stark_MHz*MHz
        return dc_stark_Hz    
    
    def saturation_intensity(self, lifetime: float, wavelength: float):
        """calculate saturation intensity

        Args:
            excited state lifetime tau in s
            wavelength in m

        Returns:
            saturation intensity in [W/m^2]
        """
        saturation_intensity = pi*(hbar*2*pi)*c/(3*lifetime*wavelength**3)
        return saturation_intensity
    
    def scattering_rate_sat(self, intensity: float, detuning: float, linewidth: float, wavelength: float=461*nm):
        """compute scattering rate given beam intensity, detuning and linewidth

        Args:
            intensity (float or np array): intensity in [W/m^2].
            detuning (float): detuning in [Hz].
            linewidth (float): linewidth in [Hz].

        Returns:
            scattering_rate (float or np array): scattering rate in [Hz].
        """
        s0 = self.saturation_intensity(1/linewidth, wavelength)
        s = intensity/s0
        scattering_rate = (linewidth/2)*(s/(1 + s + (2*detuning/linewidth)**2))
        return scattering_rate


def main():

    beam_power = 30*mW 
    beam_waist = 18*um
    uv_intensity = GaussianBeam(beam_power, beam_waist).get_intensity()
    rabi_freq = AtomLightInteraction().calc_rydberg_rabi_freq(61, uv_intensity, j_e=1)
    print("Rydberg rabi", rabi_freq/(2*pi*MHz), " MHz")

    gamma=2*pi*32*MHz
    s0 = AtomLightInteraction().saturation_intensity(1/gamma, 461*nm)
    print("s0:", s0, " W/m^2")


if __name__ == "__main__":
    main()
